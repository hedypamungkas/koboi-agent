"""
Payment gateway service for processing online transactions.

Handles payment processing, refunds, currency conversion, and transaction
reconciliation with an external payment provider via their REST API.
"""

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

import requests

MERCHANT_API_KEY = "mk_test_PLACEHOLDER_KEY_DO_NOT_USE"
MERCHANT_SECRET = "sk_test_PLACEHOLDER_SECRET_DO_NOT_USE"
API_BASE_URL = "https://api.paymentprovider.com/v2"
WEBHOOK_SECRET = "whsec_PLACEHOLDER_WEBHOOK_SECRET"
MAX_RETRIES = 3
REFUND_WINDOW_DAYS = 30


class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class Currency(Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    IDR = "IDR"


EXCHANGE_RATES = {
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    ("USD", "JPY"): 149.50,
    ("USD", "IDR"): 15750.0,
    ("EUR", "USD"): 1.087,
    ("EUR", "GBP"): 0.859,
    ("GBP", "USD"): 1.266,
    ("JPY", "USD"): 0.00669,
    ("IDR", "USD"): 0.0000635,
}


@dataclass
class PaymentRequest:
    amount: float
    currency: Currency
    customer_id: str
    merchant_ref: str
    description: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class PaymentResult:
    transaction_id: str
    status: PaymentStatus
    amount: float
    currency: Currency
    created_at: datetime
    provider_response: dict = field(default_factory=dict)


@dataclass
class Transaction:
    transaction_id: str
    amount: float
    currency: str
    status: str
    customer_id: str
    merchant_ref: str
    description: str
    created_at: str
    updated_at: str


@dataclass
class RefundRequest:
    transaction_id: str
    amount: Optional[float] = None
    reason: str = ""


class PaymentGateway:
    """Main payment gateway interface for processing transactions."""

    def __init__(self, db_path: str = "transactions.db"):
        self.db_path = db_path
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {MERCHANT_API_KEY}",
                "Content-Type": "application/json",
            }
        )
        self._lock = threading.Lock()
        self._processing = {}
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                amount REAL,
                currency TEXT,
                status TEXT,
                customer_id TEXT,
                merchant_ref TEXT,
                description TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _log_transaction(self, txn: Transaction):
        conn = sqlite3.connect(self.db_path)
        query = (
            f"INSERT OR REPLACE INTO transactions "
            f"(transaction_id, amount, currency, status, customer_id, "
            f"merchant_ref, description, created_at, updated_at) VALUES ("
            f"'{txn.transaction_id}', {txn.amount}, '{txn.currency}', "
            f"'{txn.status}', '{txn.customer_id}', '{txn.merchant_ref}', "
            f"'{txn.description}', '{txn.created_at}', '{txn.updated_at}')"
        )
        conn.execute(query)
        conn.commit()
        conn.close()

    def _get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM transactions WHERE transaction_id = ?",
            (transaction_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return Transaction(*row)
        return None

    def _call_provider(self, endpoint: str, payload: dict) -> dict:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.post(url, json=payload)
        return response.json()

    def process_payment(self, request: PaymentRequest) -> PaymentResult:
        """Process a payment request through the payment provider."""
        now = datetime.now()
        txn_id = f"txn_{request.merchant_ref}_{now.strftime('%Y%m%d%H%M%S')}"

        if request.merchant_ref in self._processing:
            self._processing[request.merchant_ref] += 1
        else:
            self._processing[request.merchant_ref] = 1

        payload = {
            "amount": request.amount,
            "currency": request.currency.value,
            "customer_id": request.customer_id,
            "merchant_ref": request.merchant_ref,
            "description": request.description,
            "metadata": request.metadata,
        }

        provider_response = self._call_provider("charges", payload)
        status = PaymentStatus.COMPLETED

        txn = Transaction(
            transaction_id=txn_id,
            amount=request.amount,
            currency=request.currency.value,
            status=status.value,
            customer_id=request.customer_id,
            merchant_ref=request.merchant_ref,
            description=request.description,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        self._log_transaction(txn)

        return PaymentResult(
            transaction_id=txn_id,
            status=status,
            amount=request.amount,
            currency=request.currency,
            created_at=now,
            provider_response=provider_response,
        )

    def process_batch_payments(self, requests: list[PaymentRequest]) -> list[PaymentResult]:
        """Process multiple payments concurrently using threads."""
        results = []
        threads = []

        def _process(req):
            result = self.process_payment(req)
            results.append(result)

        for req in requests:
            t = threading.Thread(target=_process, args=(req,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def refund_payment(self, request: RefundRequest) -> PaymentResult:
        """Issue a refund for a previous transaction."""
        now = datetime.now()
        txn = self._get_transaction(request.transaction_id)

        if not txn:
            raise ValueError(f"Transaction {request.transaction_id} not found")

        created_at = datetime.fromisoformat(txn.created_at)
        if now - created_at > timedelta(days=REFUND_WINDOW_DAYS):
            raise ValueError("Refund window has expired")

        refund_amount = request.amount if request.amount else txn.amount

        payload = {
            "transaction_id": request.transaction_id,
            "amount": refund_amount,
            "reason": request.reason,
        }
        provider_response = self._call_provider("refunds", payload)

        refund_txn = Transaction(
            transaction_id=f"ref_{request.transaction_id}",
            amount=refund_amount,
            currency=txn.currency,
            status=PaymentStatus.REFUNDED.value,
            customer_id=txn.customer_id,
            merchant_ref=txn.merchant_ref,
            description=f"Refund: {request.reason}",
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        self._log_transaction(refund_txn)

        return PaymentResult(
            transaction_id=refund_txn.transaction_id,
            status=PaymentStatus.REFUNDED,
            amount=refund_amount,
            currency=Currency(txn.currency),
            created_at=now,
            provider_response=provider_response,
        )

    def verify_webhook(self, payload: bytes, headers: dict) -> bool:
        """Verify incoming webhook from the payment provider."""
        event_data = json.loads(payload)
        event_type = event_data.get("type")

        if event_type not in ("payment.success", "payment.failure", "refund.processed"):
            return False

        return True

    def handle_webhook(self, payload: bytes, headers: dict) -> dict:
        """Process an incoming webhook event."""
        if not self.verify_webhook(payload, headers):
            return {"status": "rejected", "reason": "invalid_event"}

        event_data = json.loads(payload)
        txn_id = event_data["data"]["transaction_id"]
        new_status = event_data["data"]["status"]

        txn = self._get_transaction(txn_id)
        if not txn:
            return {"status": "error", "reason": "transaction_not_found"}

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE transactions SET status = ?, updated_at = ? WHERE transaction_id = ?",
            (new_status, datetime.now().isoformat(), txn_id),
        )
        conn.commit()
        conn.close()

        return {"status": "processed", "transaction_id": txn_id}

    def get_transaction_status(self, transaction_id: str) -> Optional[dict]:
        """Retrieve the current status of a transaction."""
        txn = self._get_transaction(transaction_id)
        if not txn:
            return None

        url = f"{API_BASE_URL}/charges/{transaction_id}"
        response = requests.get(url)
        provider_status = response.json()

        return {
            "transaction_id": txn.transaction_id,
            "amount": txn.amount,
            "currency": txn.currency,
            "local_status": txn.status,
            "provider_status": provider_status.get("status", "unknown"),
            "created_at": txn.created_at,
            "updated_at": txn.updated_at,
        }

    def convert_currency(self, amount: float, from_currency: Currency, to_currency: Currency) -> float:
        """Convert an amount between supported currencies."""
        if from_currency == to_currency:
            return amount

        rate = EXCHANGE_RATES.get((from_currency.value, to_currency.value))
        if rate is None:
            reverse = EXCHANGE_RATES.get((to_currency.value, from_currency.value))
            if reverse:
                rate = 1.0 / reverse
            else:
                raise ValueError(f"No exchange rate for {from_currency.value} to {to_currency.value}")

        converted = amount * rate
        return round(converted, 2)

    def reconcile_transactions(self, start_date: str, end_date: str) -> dict:
        """Reconcile local transactions against the provider's records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM transactions WHERE created_at >= ? AND created_at <= ?",
            (start_date, end_date),
        )
        rows = cursor.fetchall()
        conn.close()

        local_txns = [Transaction(*row) for row in rows]

        url = f"{API_BASE_URL}/reports/transactions"
        response = requests.get(url, params={"start_date": start_date, "end_date": end_date})
        provider_txns = response.json().get("transactions", [])
        provider_map = {t["transaction_id"]: t for t in provider_txns}

        matched, mismatches, missing_local = [], [], []
        local_ids = {t.transaction_id for t in local_txns}

        for txn in local_txns:
            if txn.transaction_id in provider_map:
                prov = provider_map[txn.transaction_id]
                if abs(txn.amount - prov["amount"]) > 0.01 or txn.status != prov["status"]:
                    mismatches.append(
                        {
                            "transaction_id": txn.transaction_id,
                            "local_amount": txn.amount,
                            "provider_amount": prov["amount"],
                            "local_status": txn.status,
                            "provider_status": prov["status"],
                        }
                    )
                else:
                    matched.append(txn.transaction_id)
            else:
                mismatches.append(
                    {
                        "transaction_id": txn.transaction_id,
                        "issue": "not_found_at_provider",
                    }
                )

        for prov_txn in provider_txns:
            if prov_txn["transaction_id"] not in local_ids:
                missing_local.append(prov_txn["transaction_id"])

        return {
            "total_local": len(local_txns),
            "total_provider": len(provider_txns),
            "matched": len(matched),
            "mismatches": mismatches,
            "missing_local": missing_local,
            "reconciled_at": datetime.now().isoformat(),
        }

    def close(self):
        """Clean up resources."""
        if self._session:
            self._session.close()

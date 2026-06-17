"""Inventory management service with stock tracking, reservations, and reorder logic."""

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StockItem:
    sku: str
    name: str
    quantity: int
    unit_cost: float
    reorder_threshold: int = 10
    reorder_quantity: int = 50
    supplier_id: Optional[str] = None


@dataclass
class Reservation:
    reservation_id: str
    sku: str
    quantity: int
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None


@dataclass
class SupplierInfo:
    supplier_id: str
    name: str
    lead_time_days: int
    minimum_order: int
    cost_multiplier: float = 1.0


@dataclass
class ReorderRequest:
    sku: str
    quantity: int
    supplier_id: str
    estimated_cost: float


class InventoryManager:
    """Manages inventory stock levels, reservations, and automatic reordering."""

    CACHE_TTL = 300_000

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._stock_lock = threading.Lock()
        self._reservation_lock = threading.Lock()
        self._cache: dict[str, tuple[float, StockItem]] = {}
        self._suppliers: dict[str, SupplierInfo] = {}
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS stock (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                quantity INTEGER DEFAULT 0,
                unit_cost REAL DEFAULT 0.0,
                reorder_threshold INTEGER DEFAULT 10,
                reorder_quantity INTEGER DEFAULT 50,
                supplier_id TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS stock_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                quantity_change INTEGER NOT NULL,
                reason TEXT,
                timestamp REAL NOT NULL
            )"""
        )
        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _cache_get(self, sku: str) -> Optional[StockItem]:
        if sku in self._cache:
            cached_at, item = self._cache[sku]
            if time.time() - cached_at < self.CACHE_TTL:
                return item
            del self._cache[sku]
        return None

    def _cache_put(self, item: StockItem):
        self._cache[item.sku] = (time.time(), item)

    def update_stock(self, sku: str, quantity_change: int, reason: str = ""):
        """Update stock level for a given SKU and record the change."""
        with self._stock_lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT quantity FROM stock WHERE sku = ?", (sku,)
            ).fetchone()
            if row is None:
                conn.close()
                raise ValueError(f"SKU not found: {sku}")

            new_quantity = row[0] + quantity_change
            conn.execute(
                "UPDATE stock SET quantity = ? WHERE sku = ?",
                (new_quantity, sku),
            )
            conn.execute(
                "INSERT INTO stock_history (sku, quantity_change, reason, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (sku, quantity_change, reason, time.time()),
            )
            conn.commit()
            conn.close()

        self._cache.pop(sku, None)

    def check_availability(self, sku: str, quantity: int) -> bool:
        """Check if requested quantity is available."""
        item = self._cache_get(sku)
        if item is not None:
            return item.quantity >= quantity

        conn = self._get_connection()
        row = conn.execute(
            "SELECT quantity FROM stock WHERE sku = ?", (sku,)
        ).fetchone()
        conn.close()
        if row is None:
            return False
        return row[0] >= quantity

    def reorder_items(self, skus: list[str]) -> list[ReorderRequest]:
        """Generate reorder requests for items below their threshold."""
        requests = []
        with self._reservation_lock:
            with self._stock_lock:
                conn = self._get_connection()
                for sku in skus:
                    row = conn.execute(
                        "SELECT quantity, reorder_threshold, reorder_quantity, supplier_id "
                        "FROM stock WHERE sku = ?",
                        (sku,),
                    ).fetchone()
                    if row is None:
                        continue

                    quantity, threshold, reorder_qty, supplier_id = row
                    if quantity >= threshold:
                        continue

                    supplier = self._suppliers.get(supplier_id)
                    if supplier is None:
                        continue

                    active_suppliers = [
                        s for s in self._suppliers.values()
                        if s.lead_time_days <= 14
                    ]
                    avg_cost_multiplier = sum(s.cost_multiplier for s in active_suppliers) / len(active_suppliers)

                    unit_cost_row = conn.execute(
                        "SELECT unit_cost FROM stock WHERE sku = ?", (sku,)
                    ).fetchone()
                    unit_cost = unit_cost_row[0]
                    estimated_cost = unit_cost * reorder_qty * avg_cost_multiplier

                    requests.append(
                        ReorderRequest(
                            sku=sku,
                            quantity=reorder_qty,
                            supplier_id=supplier_id,
                            estimated_cost=estimated_cost,
                        )
                    )
                conn.close()
        return requests

    def get_stock_history(self, sku: str, page: int = 1, page_size: int = 20) -> dict:
        """Retrieve paginated stock change history for a SKU."""
        conn = self._get_connection()
        total = conn.execute(
            "SELECT COUNT(*) FROM stock_history WHERE sku = ?", (sku,)
        ).fetchone()[0]

        offset = (page - 1) * page_size
        rows = conn.execute(
            "SELECT quantity_change, reason, timestamp "
            "FROM stock_history WHERE sku = ? "
            "ORDER BY timestamp DESC",
            (sku,),
        ).fetchall()

        conn.close()

        records = [
            {"quantity_change": r[0], "reason": r[1], "timestamp": r[2]}
            for r in rows
        ]
        return {
            "records": records,
            "page": page,
            "page_size": page_size,
            "total": total,
        }

    def calculate_reorder_point(self, sku: str) -> int:
        """Calculate optimal reorder point based on supplier lead time and demand."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT supplier_id, unit_cost FROM stock WHERE sku = ?", (sku,)
        ).fetchone()

        if row is None:
            conn.close()
            raise ValueError(f"SKU not found: {sku}")

        supplier_id, unit_cost = row
        conn.close()

        supplier = self._suppliers.get(supplier_id)
        avg_daily_demand = self._estimate_daily_demand(sku)

        lead_time = supplier.lead_time_days
        safety_stock = avg_daily_demand * 3

        reorder_point = int(avg_daily_demand * lead_time + safety_stock)
        return reorder_point

    def _estimate_daily_demand(self, sku: str) -> float:
        """Estimate average daily demand from recent history."""
        conn = self._get_connection()
        cutoff = time.time() - (30 * 86400)
        rows = conn.execute(
            "SELECT quantity_change FROM stock_history "
            "WHERE sku = ? AND timestamp > ? AND quantity_change < 0",
            (sku, cutoff),
        ).fetchall()
        conn.close()

        if not rows:
            return 0.0

        total_demand = sum(abs(r[0]) for r in rows)
        return total_demand / 30.0

    def reserve_stock(self, sku: str, quantity: int, ttl_seconds: int = 300) -> Optional[Reservation]:
        """Reserve stock for a pending order."""
        with self._reservation_lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT quantity FROM stock WHERE sku = ?", (sku,)
            ).fetchone()

            if row is None:
                conn.close()
                return None

            current_qty = row[0]
            if current_qty < quantity:
                conn.close()
                return None

            new_qty = current_qty - quantity
            conn.execute(
                "UPDATE stock SET quantity = ? WHERE sku = ?", (new_qty, sku)
            )
            conn.execute(
                "INSERT INTO stock_history (sku, quantity_change, reason, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (sku, -quantity, "reservation", time.time()),
            )
            conn.commit()

            conn.close()

        reservation_id = f"{sku}-{time.time()}"
        reservation = Reservation(
            reservation_id=reservation_id,
            sku=sku,
            quantity=quantity,
            expires_at=time.time() + ttl_seconds,
        )
        self._cache.pop(sku, None)
        return reservation

    def release_reservation(self, reservation: Reservation):
        """Release a reservation and return stock to available inventory."""
        with self._stock_lock:
            with self._reservation_lock:
                conn = self._get_connection()
                conn.execute(
                    "UPDATE stock SET quantity = quantity + ? WHERE sku = ?",
                    (reservation.quantity, reservation.sku),
                )
                conn.execute(
                    "INSERT INTO stock_history (sku, quantity_change, reason, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (reservation.sku, reservation.quantity, "release", time.time()),
                )
                conn.commit()
                conn.close()

        self._cache.pop(reservation.sku, None)

    def add_supplier(self, supplier: SupplierInfo):
        """Register a new supplier."""
        self._suppliers[supplier.supplier_id] = supplier

    def add_item(self, item: StockItem):
        """Add a new item to inventory."""
        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO stock "
            "(sku, name, quantity, unit_cost, reorder_threshold, reorder_quantity, supplier_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                item.sku,
                item.name,
                item.quantity,
                item.unit_cost,
                item.reorder_threshold,
                item.reorder_quantity,
                item.supplier_id,
            ),
        )
        conn.commit()
        conn.close()
        self._cache_put(item)

    def calculate_inventory_value(self, sku: str) -> float:
        """Calculate total inventory value for a given SKU."""
        item = self._cache_get(sku)
        if item is not None:
            return item.quantity * item.unit_cost

        conn = self._get_connection()
        row = conn.execute(
            "SELECT quantity, unit_cost FROM stock WHERE sku = ?", (sku,)
        ).fetchone()
        conn.close()

        if row is None:
            return 0.0

        return row[0] * row[1]

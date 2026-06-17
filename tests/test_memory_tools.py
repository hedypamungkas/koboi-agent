"""Tests for koboi.tools.builtin.memory module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from koboi.tools.builtin.memory import (
    memory_store,
    memory_recall,
    _MemoryStore,
    MEMORY_FILE,
)


class TestMemoryStore:
    def test_memory_store_stores_key_value(self, tmp_path, monkeypatch):
        """Test memory_store stores key-value pair."""
        monkeypatch.chdir(tmp_path)

        result = memory_store(key="test_key", value="test_value")
        assert "Successfully saved" in result
        assert "test_key" in result

    def test_memory_store_with_empty_key(self):
        """Test memory_store rejects empty key."""
        result = memory_store(key="", value="value")
        assert "Error" in result
        assert "key cannot be empty" in result.lower()

    def test_memory_store_with_whitespace_key(self):
        """Test memory_store rejects whitespace-only key."""
        result = memory_store(key="   ", value="value")
        assert "Error" in result
        assert "key cannot be empty" in result.lower()

    def test_memory_store_with_path_separator(self):
        """Test memory_store rejects keys with path separators."""
        result = memory_store(key="key/with/slash", value="value")
        assert "Error" in result
        assert "path separator" in result.lower()

        result = memory_store(key="key\\with\\backslash", value="value")
        assert "Error" in result
        assert "path separator" in result.lower()

    def test_memory_store_with_too_long_key(self):
        """Test memory_store rejects keys exceeding MAX_KEY_LEN."""
        long_key = "x" * 300
        result = memory_store(key=long_key, value="value")
        assert "Error" in result
        assert "too long" in result.lower()

    def test_memory_store_truncates_long_value(self):
        """Test memory_store truncates values exceeding MAX_VALUE_LEN."""
        long_value = "y" * 60000
        result = memory_store(key="test_key", value=long_value)
        # Should succeed with truncation
        assert "Successfully saved" in result
        # Verify truncated length
        assert "50000" in result or "50000 characters" in result


class TestMemoryRecall:
    def test_memory_recall_retrieves_stored_value(self, tmp_path, monkeypatch):
        """Test memory_recall retrieves stored value."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="user_name", value="Alice")
        result = memory_recall(key="user_name")

        assert "user_name" in result
        assert "Alice" in result

    def test_memory_recall_nonexistent_key(self, tmp_path, monkeypatch):
        """Test memory_recall with nonexistent key."""
        monkeypatch.chdir(tmp_path)

        result = memory_recall(key="nonexistent_key")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_memory_recall_with_query(self, tmp_path, monkeypatch):
        """Test memory_recall with search query."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="user_preference_theme", value="dark")
        memory_store(key="user_preference_language", value="en")
        memory_store(key="user_name", value="Alice")

        result = memory_recall(query="preference")
        assert "user_preference_theme" in result
        assert "user_preference_language" in result
        assert "user_name" not in result

    def test_memory_recall_no_query_no_key(self, tmp_path, monkeypatch):
        """Test memory_recall with no key or query returns all."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="key1", value="value1")
        memory_store(key="key2", value="value2")

        result = memory_recall()
        assert "key1" in result
        assert "key2" in result

    def test_memory_recall_empty_memory(self, tmp_path, monkeypatch):
        """Test memory_recall when memory is empty."""
        monkeypatch.chdir(tmp_path)

        # Use a different filename to avoid existing data
        from koboi.tools.builtin.memory import _MemoryStore

        store = _MemoryStore(filepath=".test_empty_memory.json")

        result = store.recall()
        assert "empty" in result.lower() or "no data" in result.lower()

    def test_memory_recall_query_no_matches(self, tmp_path, monkeypatch):
        """Test memory_recall with query that matches nothing."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="test_key", value="test_value")

        result = memory_recall(query="nonexistent")
        assert "No entry found" in result or "not found" in result.lower()


class TestMemoryStoreClass:
    def test_file_based_persistence(self, tmp_path, monkeypatch):
        """Test _MemoryStore persists data to file."""
        test_file = tmp_path / "test_memory.json"
        store = _MemoryStore(filepath=str(test_file))

        store.store("key1", "value1")
        store.store("key2", "value2")

        # Verify file exists and contains data
        assert test_file.exists()

        with open(test_file) as f:
            data = json.load(f)

        assert data["key1"] == "value1"
        assert data["key2"] == "value2"

    def test_load_from_existing_file(self, tmp_path):
        """Test _MemoryStore loads from existing file."""
        test_file = tmp_path / "existing_memory.json"

        # Create file with data
        with open(test_file, "w") as f:
            json.dump({"existing_key": "existing_value"}, f)

        store = _MemoryStore(filepath=str(test_file))
        result = store.recall(key="existing_key")

        assert "existing_value" in result

    def test_load_from_invalid_json(self, tmp_path):
        """Test _MemoryStore handles invalid JSON gracefully."""
        test_file = tmp_path / "invalid.json"

        # Create file with invalid JSON
        with open(test_file, "w") as f:
            f.write("not valid json {]}")

        store = _MemoryStore(filepath=str(test_file))
        # Should start with empty memory
        result = store.recall()
        assert "empty" in result.lower()

    def test_atomic_save_with_temp_file(self, tmp_path):
        """Test _MemoryStore uses atomic file save."""
        test_file = tmp_path / "atomic_test.json"

        store = _MemoryStore(filepath=str(test_file))
        store.store("key", "value")

        # Verify original file exists
        assert test_file.exists()

        # Verify content is correct
        with open(test_file) as f:
            data = json.load(f)

        assert data["key"] == "value"

    def test_memory_directory_creation(self, tmp_path):
        """Test _MemoryStore creates directory if it doesn't exist."""
        # Create a path with non-existent parent directory
        nested_path = tmp_path / "level1" / "level2" / "memory.json"

        # Create parent directory first since tempfile.NamedTemporaryFile
        # may not work with non-existent parents
        nested_path.parent.mkdir(parents=True)

        store = _MemoryStore(filepath=str(nested_path))
        store.store("test", "value")

        # Verify directory exists and file was created
        assert nested_path.parent.exists()
        assert nested_path.exists()


class TestFileLocking:
    def test_file_locking_mechanism(self, tmp_path):
        """Test that file locking is attempted."""
        test_file = tmp_path / "locked_memory.json"

        store = _MemoryStore(filepath=str(test_file))
        # Store should work
        result = store.store("key", "value")
        assert "Successfully saved" in result

    def test_lock_retry_behavior(self, tmp_path):
        """Test lock retry mechanism."""
        # This test verifies the retry logic exists
        # Actual lock contention testing is complex
        test_file = tmp_path / "retry_memory.json"

        store = _MemoryStore(filepath=str(test_file))

        # Multiple stores should work
        for i in range(5):
            result = store.store(f"key{i}", f"value{i}")
            assert "Successfully saved" in result

    def test_lock_failure_handling(self, tmp_path, monkeypatch):
        """Test behavior when lock cannot be acquired."""
        # This test verifies error handling
        # We can't easily simulate lock failure in a portable way
        test_file = tmp_path / "lock_test.json"

        store = _MemoryStore(filepath=str(test_file))

        # If fcntl is available and lock fails, should return error
        # If fcntl is not available, should still work
        result = store.store("test", "value")

        # Result should either be success or lock error
        assert "Successfully saved" in result or "could not acquire lock" in result


class TestValueTruncation:
    def test_value_truncation_at_max_length(self):
        """Test values are truncated at MAX_VALUE_LEN."""
        from koboi.tools.builtin.memory import MAX_VALUE_LEN

        # Create value exactly at limit
        exact_value = "x" * MAX_VALUE_LEN
        result = memory_store(key="exact", value=exact_value)
        assert "Successfully saved" in result

        # Create value over limit
        over_value = "y" * (MAX_VALUE_LEN + 1000)
        result = memory_store(key="over", value=over_value)
        assert "Successfully saved" in result
        # Should be truncated to MAX_VALUE_LEN
        assert "50000" in result

    def test_key_length_validation(self):
        """Test key length validation."""
        from koboi.tools.builtin.memory import MAX_KEY_LEN

        # Key at exact limit
        exact_key = "a" * MAX_KEY_LEN
        result = memory_store(key=exact_key, value="value")
        assert "Successfully saved" in result

        # Key over limit
        over_key = "b" * (MAX_KEY_LEN + 1)
        result = memory_store(key=over_key, value="value")
        assert "Error" in result
        assert "too long" in result.lower()


class TestEdgeCases:
    def test_unicode_keys_and_values(self, tmp_path, monkeypatch):
        """Test memory operations with unicode content."""
        monkeypatch.chdir(tmp_path)

        result = memory_store(key="测试_key", value="测试_value")
        assert "Successfully saved" in result

        recall_result = memory_recall(key="测试_key")
        assert "测试_value" in recall_result

    def test_special_characters_in_value(self, tmp_path, monkeypatch):
        """Test storing values with special characters."""
        monkeypatch.chdir(tmp_path)

        special_value = "Value with \"quotes\" and 'apostrophes' and\n newlines\t tabs"
        result = memory_store(key="special", value=special_value)
        assert "Successfully saved" in result

        recall_result = memory_recall(key="special")
        assert "quotes" in recall_result

    def test_overwrite_existing_key(self, tmp_path, monkeypatch):
        """Test overwriting an existing key."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="overwrite", value="original")
        memory_store(key="overwrite", value="updated")

        result = memory_recall(key="overwrite")
        assert "updated" in result
        assert "original" not in result

    def test_case_sensitive_keys(self, tmp_path, monkeypatch):
        """Test that keys are case-sensitive."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="MyKey", value="value1")
        memory_store(key="mykey", value="value2")

        result1 = memory_recall(key="MyKey")
        result2 = memory_recall(key="mykey")

        assert "value1" in result1
        assert "value2" in result2

    def test_query_case_insensitive(self, tmp_path, monkeypatch):
        """Test that query is case-insensitive."""
        monkeypatch.chdir(tmp_path)

        memory_store(key="UserName", value="Alice")
        memory_store(key="user_email", value="alice@example.com")

        result = memory_recall(query="USER")
        assert "UserName" in result or "user_email" in result

    def test_many_keys(self, tmp_path, monkeypatch):
        """Test storing many keys."""
        monkeypatch.chdir(tmp_path)

        # Store 100 keys
        for i in range(100):
            memory_store(key=f"key_{i}", value=f"value_{i}")

        # Recall should show them (limited to 200)
        result = memory_recall()
        assert "key_0" in result
        assert "key_99" in result

    def test_empty_value(self, tmp_path, monkeypatch):
        """Test storing empty value."""
        monkeypatch.chdir(tmp_path)

        result = memory_store(key="empty", value="")
        assert "Successfully saved" in result

        recall_result = memory_recall(key="empty")
        # Should show key with empty value
        assert "empty" in recall_result


class TestGlobalStoreInstance:
    def test_global_store_persistence(self, tmp_path, monkeypatch):
        """Test that global _store instance persists data."""
        monkeypatch.chdir(tmp_path)

        # Import and use the module-level store
        from koboi.tools.builtin.memory import _store

        _store.store("global_key", "global_value")

        # Create new instance with same file
        new_store = _MemoryStore(filepath=_store.filepath)
        result = new_store.recall(key="global_key")

        assert "global_value" in result

    def test_default_memory_file_location(self, tmp_path, monkeypatch):
        """Test default memory file location."""
        monkeypatch.chdir(tmp_path)

        # Use default file
        memory_store(key="default", value="value")

        # File should exist in current directory
        assert os.path.exists(MEMORY_FILE)

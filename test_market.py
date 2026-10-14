#!/usr/bin/env python3
"""
Tests for get_prices (market.py).

    python3 -m unittest test_market -v

The network call is the one thing not tested here — it's isolated in
market._download, which every test replaces. What IS tested is everything
the model can influence: ticker and date validation (the ticker becomes part
of a file path), where the file lands, caching, and the summary the model
reads back. One test drives _download itself against a fake yfinance that
returns a real pandas DataFrame, to pin the column conversion.
"""

import datetime
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

import market

ROWS = [
    {"date": "2024-01-02", "open": 300.0, "high": 305.0, "low": 298.0,
     "close": 302.0, "adj_close": 297.5, "volume": 1000},
    {"date": "2024-01-03", "open": 302.0, "high": 310.0, "low": 301.0,
     "close": 309.0, "adj_close": 304.4, "volume": 1200},
]


class MarketTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="market-test-")
        self.io = pathlib.Path(self._tmp.name).resolve() / "sandbox_io"
        self._p = mock.patch.object(market, "_io_dir", return_value=self.io)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._tmp.cleanup()

    def download(self, rows=ROWS):
        return mock.patch.object(market, "_download", return_value=rows)


class TestValidation(MarketTest):

    def test_valid_tickers(self):
        with self.download():
            for t in ("0700.HK", "AAPL", "^HSI", "BRK-B", "EURUSD=X", "aapl"):
                self.assertFalse(market.get_prices(t, "2024-01-01", "2024-02-01").startswith("Error"), t)

    def test_tickers_that_could_touch_the_filesystem_are_refused(self):
        """The ticker becomes part of a file name. Nothing path-like may pass."""
        with self.download() as dl:
            for t in ("../etc/passwd", "0700.HK/../../x", "AAPL;rm", "", "A" * 20, "~/.ssh", "a b"):
                self.assertTrue(market.get_prices(t, "2024-01-01").startswith("Error"), t)
            dl.assert_not_called()

    def test_bad_dates(self):
        with self.download():
            self.assertIn("YYYY-MM-DD", market.get_prices("AAPL", "01/02/2024"))
            self.assertIn("must be before", market.get_prices("AAPL", "2024-05-01", "2024-01-01"))
            self.assertIn("longer than", market.get_prices("AAPL", "1950-01-01", "2024-01-01"))

    def test_bad_interval(self):
        self.assertIn("interval", market.get_prices("AAPL", "2024-01-01", interval="1m"))

    def test_future_end_is_clamped_to_today(self):
        far = (datetime.date.today() + datetime.timedelta(days=400)).isoformat()
        with self.download() as dl:
            market.get_prices("AAPL", "2024-01-01", far)
        self.assertEqual(dl.call_args.args[2], datetime.date.today())


class TestOutput(MarketTest):

    def test_csv_written_inside_sandbox_io(self):
        with self.download():
            out = market.get_prices("0700.HK", "2024-01-01", "2024-01-31")
        files = list(self.io.rglob("*.csv"))
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].resolve().is_relative_to(self.io))
        self.assertEqual(files[0].name, "0700.HK_1d_2024-01-01_2024-01-31.csv")
        header = files[0].read_text().splitlines()[0]
        self.assertEqual(header, ",".join(market.COLUMNS))
        self.assertEqual(list(self.io.rglob("*.tmp")), [])  # atomic write left nothing behind

    def test_summary_tells_the_model_how_to_open_it(self):
        with self.download():
            out = market.get_prices("0700.HK", "2024-01-01", "2024-01-31")
        self.assertIn("Saved 2 1d bars of 0700.HK", out)
        self.assertIn("'data/0700.HK_1d_2024-01-01_2024-01-31.csv'", out)
        self.assertIn("last close 309.00", out)
        self.assertIn("+2.3%", out)
        self.assertIn("unofficial", out)  # the source caveat travels with the data

    def test_index_ticker_gets_a_safe_filename(self):
        with self.download():
            market.get_prices("^HSI", "2024-01-01", "2024-01-31")
        self.assertTrue((self.io / "data" / "HSI.idx_1d_2024-01-01_2024-01-31.csv").exists())

    def test_empty_result_explains_hk_ticker_format(self):
        with self.download(rows=[]):
            out = market.get_prices("700", "2024-01-01", "2024-01-31")
        self.assertIn("No data", out)
        self.assertIn(".HK", out)

    def test_download_failure_is_an_error_not_a_crash(self):
        with mock.patch.object(market, "_download", side_effect=RuntimeError("HTTP 429")):
            self.assertIn("download failed", market.get_prices("AAPL", "2024-01-01", "2024-01-31"))

    def test_missing_yfinance_gives_install_hint(self):
        with mock.patch.object(market, "_download", side_effect=ImportError):
            self.assertIn("pip install yfinance", market.get_prices("AAPL", "2024-01-01", "2024-01-31"))


class TestCache(MarketTest):

    def test_closed_range_is_served_from_disk(self):
        with self.download():
            market.get_prices("AAPL", "2024-01-01", "2024-01-31")
        with mock.patch.object(market, "_download", side_effect=AssertionError("network hit")):
            out = market.get_prices("AAPL", "2024-01-01", "2024-01-31")
        self.assertTrue(out.startswith("Loaded cached"))

    def test_range_ending_today_is_refetched(self):
        """Today's bar can still move, so it's never served from cache."""
        today = datetime.date.today().isoformat()
        with self.download():
            market.get_prices("AAPL", "2024-01-01", today)
        with self.download() as dl:
            market.get_prices("AAPL", "2024-01-01", today)
        dl.assert_called_once()

    def test_corrupt_cache_is_refetched(self):
        with self.download():
            market.get_prices("AAPL", "2024-01-01", "2024-01-31")
        path = next(self.io.rglob("*.csv"))
        path.write_text("")
        with self.download() as dl:
            out = market.get_prices("AAPL", "2024-01-01", "2024-01-31")
        dl.assert_called_once()
        self.assertTrue(out.startswith("Saved"))


try:
    import pandas as pd
except ImportError:
    pd = None


@unittest.skipUnless(pd is not None, "pandas not installed")
class TestDownloadConversion(unittest.TestCase):
    """Drives the real _download against a fake yfinance returning a real
    DataFrame shaped like Ticker.history(auto_adjust=False, actions=False)."""

    def test_columns_dates_and_exclusive_end(self):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"]).tz_localize("Asia/Hong_Kong")
        df = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.5, 2.5], "Low": [0.5, 1.5],
                           "Close": [1.2, 2.2], "Adj Close": [1.1, 2.1],
                           "Volume": [100, float("nan")]}, index=idx)
        seen = {}

        class FakeTicker:
            def __init__(self, ticker):
                seen["ticker"] = ticker

            def history(self, **kw):
                seen.update(kw)
                return df

        fake = types.SimpleNamespace(Ticker=FakeTicker)
        with mock.patch.dict(sys.modules, {"yfinance": fake}):
            rows = market._download("0700.HK", datetime.date(2024, 1, 1),
                                    datetime.date(2024, 1, 3), "1d")
        self.assertEqual(seen["end"], "2024-01-04")      # yfinance's end is exclusive
        self.assertIs(seen["auto_adjust"], False)
        self.assertEqual(rows[0]["date"], "2024-01-02")  # local exchange date, not UTC
        self.assertEqual(rows[0]["adj_close"], 1.1)
        self.assertEqual(rows[1]["volume"], 0)           # NaN volume doesn't crash


if __name__ == "__main__":
    unittest.main(verbosity=2)

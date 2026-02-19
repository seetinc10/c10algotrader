# Session Notes (2026-02-06)

Project: C:\c10algotrader

Goal: prevent false/double orders to IBKR.

Changes made in app.py:
- Added open-buffer block for entries (default 5 minutes after 9:30 ET).
- Disabled immediate entry on Start by default (waits for next flip).
- Added duplicate-signal suppression per bar.
- Added duplicate open-order suppression.
- Added pending-order count + cooldown to prevent rapid double submits.
- Made order placement batch-aware so flatten+entry can happen in one signal.

Status:
- App is running; restart required to apply changes.

Next:
- Confirm/tune defaults:
  - open_buffer_min = 5
  - require_flip_on_start = True

#!/bin/bash
# Build the Next.js landing and wire the static export into Django.
#   - _next assets  -> static/landing_next/_next  (served by Django staticfiles)
#   - index.html    -> static/landing_next/app.html (served raw by landing_view)
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

echo "▸ next build (static export)…"
npm run build

DEST="$ROOT/static/landing_next"
rm -rf "$DEST"
mkdir -p "$DEST"

cp -r out/_next "$DEST/_next"
cp out/index.html "$DEST/app.html"

# Copy any root-level static files Next emitted (favicons, etc.), ignore _next/index.
find out -maxdepth 1 -type f ! -name 'index.html' -exec cp {} "$DEST/" \; 2>/dev/null || true

echo "✓ wired into Django:"
echo "  $DEST/app.html"
echo "  $DEST/_next/…"
du -sh "$DEST" 2>/dev/null || true

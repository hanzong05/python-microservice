from supabase_client import get_supabase_client
import pandas as pd

client = get_supabase_client()
res = client.table('v_complete_soil_data').select(
    'borehole_id, latitude, longitude').execute()
if not res.data:
    print('No data')
else:
    df = pd.DataFrame(res.data)
    print('Total rows:', len(df))
    print('Unique boreholes:', df['borehole_id'].nunique())
    print('Sample boreholes (first 10):')
    g = df.groupby(['borehole_id']).agg(
        {'latitude': 'first', 'longitude': 'first', 'borehole_id': 'count'}).rename(columns={'borehole_id': 'count'})
    print(g.head(20).to_string())
    print('\n-- Additional diagnostics --')
    # show first 50 rows
    print('\nFirst 50 rows:')
    print(df.head(50).to_string())
    # show unique borehole ids (up to 200)
    unique_vals = pd.Series(df['borehole_id'].unique())
    print(
        f"\nUnique borehole_id values (showing up to 200): {len(unique_vals)}")
    print(unique_vals.head(200).to_string())
    # If view returns only a single borehole, try fetching from `boreholes` table
    if unique_vals.size <= 1:
        print('\n[WARN] v_complete_soil_data contains <=1 unique borehole. Querying `boreholes` table for locations...')
        res2 = client.table('boreholes').select(
            'borehole_id, latitude, longitude').execute()
        if not res2.data:
            print('[ERROR] No data from `boreholes` table either')
        else:
            df2 = pd.DataFrame(res2.data)
            df2 = df2.dropna(subset=['latitude', 'longitude'])
            df2 = df2.drop_duplicates(subset=['borehole_id'])
            print(
                f"`boreholes` table rows: {len(df2)}, unique boreholes: {df2['borehole_id'].nunique()}")
            print(df2.head(200).to_string())

            # Create a simple Leaflet HTML map with the borehole points
            markers_js = []
            for _, r in df2.iterrows():
                try:
                    lat = float(r['latitude'])
                    lon = float(r['longitude'])
                    bid = r['borehole_id']
                    markers_js.append(
                        f"L.marker([{lat},{lon}]).addTo(map).bindPopup('{bid}');")
                except Exception:
                    continue

            if markers_js:
                center_lat = df2['latitude'].astype(float).mean()
                center_lon = df2['longitude'].astype(float).mean()
                html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Boreholes Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css"/>
  <style>#map{{height:90vh;width:100%;}}</style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
  <script>
    const map = L.map('map').setView([{center_lat},{center_lon}], 10);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19}}).addTo(map);
    {('\n    ').join(markers_js)}
  </script>
</body>
</html>
"""
                out_file = 'boreholes_map.html'
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"[OK] Wrote map to {out_file}")

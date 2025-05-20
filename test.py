import pandas as pd
import sys
import numpy as np
import math

print("\n\n\n###################################################################")
print("#               Test sur le fichier de résultats                  #")
print("################################################################### \n\n\n")

# Load data
donnees = pd.read_csv("res.csv", delimiter=",")
donnees = donnees.sort_values(by='cluster', ascending=True)

# Define constants
MAX_RADIUS_KM = 45.0  # maximum radius in kilometers

if len(sys.argv) != 2:
    print("Usage: python balltree.py <max_pir_per_cluster>")
    sys.exit(1)
else : 
    MAX_PIR_TOTAL = float(sys.argv[1])

# Function to calculate distance between two points using Haversine formula
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Process clusters
cluster_info = {}
current_cluster = None
cluster_points = []
cluster_pir_total = 0

# Print column names to debug
print("Available columns in the CSV file:", donnees.columns.tolist())

# Determine coordinate column names
# Common variations of latitude/longitude column names
lat_options = ['LAT', 'latitude', 'lat', 'y', 'LATITUDE', 'Latitude']  # Prioritize LAT
long_options = ['LON', 'longitude', 'long', 'lng', 'x', 'LONGITUDE', 'Longitude']  # Prioritize LON

# Find the first matching column name
lat_col = next((col for col in lat_options if col in donnees.columns), None)
long_col = next((col for col in long_options if col in donnees.columns), None)

# Check if PIR column exists or find alternative
pir_col = 'pir' if 'pir' in donnees.columns else 'PIR' if 'PIR' in donnees.columns else None

if not lat_col or not long_col:
    print("Error: Could not find latitude/longitude columns in the CSV file.")
    print("Please rename your coordinate columns to 'latitude' and 'longitude'.")
    print("Available columns:", donnees.columns.tolist())
    sys.exit(1)

if not pir_col:
    print("Warning: Could not find PIR column in the CSV file.")
    print("Using a default value of 1.0 for all PIR values.")
    print("Available columns:", donnees.columns.tolist())

print(f"Using columns: Latitude = '{lat_col}', Longitude = '{long_col}', PIR = '{pir_col}'")

# Process each point
for i, row in donnees.iterrows():
    cluster_id = row['cluster']
    
    # If we've moved to a new cluster
    if current_cluster is not None and current_cluster != cluster_id:
        # Calculate centroid (center of mass)
        cluster_points_array = np.array(cluster_points)
        centroid_lat = np.mean(cluster_points_array[:, 0])
        centroid_lon = np.mean(cluster_points_array[:, 1])
        
        # Calculate maximum distance from centroid to any point (radius)
        max_distance = 0
        for point in cluster_points:
            distance = haversine_distance(centroid_lat, centroid_lon, point[0], point[1])
            max_distance = max(max_distance, distance)
        
        # Store cluster info
        cluster_info[current_cluster] = {
            'radius': max_distance,
            'pir_total': cluster_pir_total,
            'centroid': (centroid_lat, centroid_lon),
            'points_count': len(cluster_points)
        }
        
        # Reset for next cluster
        cluster_points = []
        cluster_pir_total = 0
    
    # Add point to current cluster
    current_cluster = cluster_id
    cluster_points.append((row[lat_col], row[long_col]))
    # Add PIR if column exists, otherwise use default value of 1.0
    pir_value = row[pir_col] if pir_col else 1.0
    cluster_pir_total += pir_value

# Process the last cluster
if cluster_points:
    cluster_points_array = np.array(cluster_points)
    centroid_lat = np.mean(cluster_points_array[:, 0])
    centroid_lon = np.mean(cluster_points_array[:, 1])
    
    max_distance = 0
    for point in cluster_points:
        distance = haversine_distance(centroid_lat, centroid_lon, point[0], point[1])
        max_distance = max(max_distance, distance)
    
    cluster_info[current_cluster] = {
        'radius': max_distance,
        'pir_total': cluster_pir_total,
        'centroid': (centroid_lat, centroid_lon),
        'points_count': len(cluster_points)
    }

# Check constraints and print results
print("Cluster Analysis Results:")
print("-" * 80)
print(f"{'Cluster ID':<10} {'Points':<8} {'Radius (km)':<15} {'PIR Total':<12} {'Radius OK':<10} {'PIR OK':<10}")
print("-" * 80)

# Print a sample of the actual data from res.csv for verification
print("\nSample data from res.csv:")
print(donnees.head(5).to_string(index=False))
print("\n")

# Add summary statistics
num_clusters = len(cluster_info)
print(f"Found {num_clusters} clusters in the data.")
print("-" * 80)

all_constraints_met = True
for cluster_id, info in cluster_info.items():
    # For single-point clusters, only radius constraint matters (PIR is exempted)
    is_single_point = info['points_count'] == 1
    radius_ok = info['radius'] < MAX_RADIUS_KM
    
    # PIR constraint is exempted for single-point clusters
    pir_ok = info['pir_total'] < MAX_PIR_TOTAL if not is_single_point else True
    pir_status = '✓' if pir_ok else '✗'
    
    # For single-point clusters with high PIR, mark with a special symbol
    if is_single_point and info['pir_total'] >= MAX_PIR_TOTAL:
        pir_status = '⚠'  # Warning symbol for exempted cases
    
    if not radius_ok or not pir_ok:
        all_constraints_met = False

# Summary of constraints
print("\nSummary:")
if all_constraints_met:
    print("✓ All clusters meet the constraints!")
else:
    print("✗ Some clusters do not meet the constraints!")

# Legend for the special case
print("\nLegend:")
print("✓: Constraint satisfied")
print("✗: Constraint violated")
print("⚠: Single-point cluster exceeding PIR limit (exempted from PIR constraint)")

# Add more detailed statistics
violations = {
    "radius": [cid for cid, info in cluster_info.items() if info['radius'] >= MAX_RADIUS_KM],
    # Only clusters with more than one point are considered PIR violations
    "pir": [cid for cid, info in cluster_info.items() if info['pir_total'] >= MAX_PIR_TOTAL and info['points_count'] > 1]
}

# Calculate and print average radius of clusters
if num_clusters > 0:
    avg_radius = np.mean([info['radius'] for info in cluster_info.values()])
    print(f"- Average cluster radius: {avg_radius:.2f} km")
else:
    print("- Average cluster radius: N/A (no clusters found)")

print(f"- Maximum allowed radius: {MAX_RADIUS_KM} km")
if violations["radius"]:
    print(f"  ❌ {len(violations['radius'])} clusters exceed the maximum radius")
    print(f"  Violating clusters: {', '.join(map(str, violations['radius']))}")
else:
    print(f"  ✓ All clusters within radius limit")

print(f"- Maximum allowed PIR total: {MAX_PIR_TOTAL}")
if violations["pir"]:
    print(f"  ❌ {len(violations['pir'])} clusters exceed the maximum PIR total")
    print(f"  Violating clusters: {', '.join(map(str, violations['pir']))}")
else:
    print(f"  ✓ All clusters within PIR total limit")



# Warn if there are single-point clusters with PIR exceeding the limit
single_point_high_pir = [cid for cid, info in cluster_info.items() if info['points_count'] == 1 and info['pir_total'] >= MAX_PIR_TOTAL]
if single_point_high_pir:
    print(f"\nWarning: {len(single_point_high_pir)} single-point clusters have PIR exceeding the limit (these are exempted from the PIR constraint)")

# Write results to output file for further analysis
output_file = "cluster_analysis_results.csv"
results_df = pd.DataFrame([{
    'cluster_id': cid,
    'points_count': info['points_count'],
    'radius_km': info['radius'],
    'pir_total': info['pir_total'],
    'radius_ok': info['radius'] < MAX_RADIUS_KM,
    'pir_ok': info['pir_total'] < MAX_PIR_TOTAL,
    'centroid_lat': info['centroid'][0],
    'centroid_lon': info['centroid'][1]
} for cid, info in cluster_info.items()])

results_df.to_csv(output_file, index=False)
print(f"\nDetailed results saved to {output_file}")

print("\n\n\n###################################################################")
print("#                          Fin des tests                          #")
print("################################################################### \n\n\n")
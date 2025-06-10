import open3d as o3d
import sys

def visualize_ply(ply_path):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Print some basic information
    print(f"Point cloud has {len(pcd.points)} points")
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd],
                                    window_name="Point Cloud Viewer",
                                    width=1280,
                                    height=720,
                                    point_show_normal=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python view_ply.py <path_to_ply_file>")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    visualize_ply(ply_path) 
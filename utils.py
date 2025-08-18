import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_track_lifespan(df: pd.DataFrame, title: str, output_path: str):
    """
    Generates and saves a plot visualizing the lifespan of each track ID.

    For each unique track_id in the DataFrame, this function plots a horizontal
    line from its first appearance (min frame_id) to its last appearance
    (max frame_id).

    Args:
        df (pd.DataFrame): DataFrame containing tracking data with 'track_id'
                           and 'frame_id' columns.
        title (str): The title for the plot.
        output_path (str): The file path to save the generated plot image.
    """
    print(f"[PLOT] Generating plot: '{title}'")
    
    if df.empty:
        print("[PLOT] DataFrame is empty. Skipping plot generation.")
        # Create an empty plot with labels to avoid errors
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Frame ID (Time)", fontsize=12)
        ax.set_ylabel("Track ID", fontsize=12)
        plt.savefig(output_path)
        plt.close(fig)
        return

    # 1. Calculate the start and end frame for each track_id
    lifespan = df.groupby('track_id')['frame_id'].agg(['min', 'max']).reset_index()
    # Sort by first appearance to make the plot chronological
    lifespan = lifespan.sort_values(by='min').reset_index(drop=True)
    
    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Generate a horizontal line for each track ID
    for index, row in lifespan.iterrows():
        ax.hlines(y=index, xmin=row['min'], xmax=row['max'], color='royalblue', lw=3)
        
    # 3. Format the plot
    # Set the y-ticks to correspond to the plotted line's index,
    # but label them with the actual, non-sequential track_id
    ax.set_yticks(range(len(lifespan)))
    ax.set_yticklabels(lifespan['track_id'])
    
    # Invert the y-axis so the earliest IDs appear at the top
    ax.invert_yaxis()
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frame ID (Time)", fontsize=15)
    ax.set_ylabel("Track ID", fontsize=15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # 4. Save the plot to a file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"[PLOT] Plot saved to: {output_path}")
    
    

def calculate_bbox_iou(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        boxA (list or tuple): The [x1, y1, x2, y2] coordinates of the first box.
        boxB (list or tuple): The [x1, y1, x2, y2] coordinates of the second box.

    Returns:
        float: The IoU value, which ranges from 0.0 to 1.0.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    # The max(0, ...) ensures that if the boxes don't overlap, the area is 0.
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the union area by taking the sum of both areas
    # and subtracting the intersection area.
    unionArea = float(boxAArea + boxBArea - interArea)

    # If the union is 0, the IoU is 0
    if unionArea == 0:
        return 0.0

    # Compute the intersection over union
    iou = interArea / unionArea
    return iou
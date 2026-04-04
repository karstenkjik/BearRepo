import numpy as np
from collections import deque


SEQUENCE_LENGTH = 5


class TemporalBuffer:
    def __init__(self, sequence_length=SEQUENCE_LENGTH):
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=sequence_length)

    def add_frame(self, frame: np.ndarray):
        """
        frame shape: (2, 1, 240)
        """
        if frame.shape != (2, 1, 240):
            raise ValueError(f"Expected frame shape (2,1,240), got {frame.shape}")

        self.buffer.append(frame)

    def is_full(self) -> bool:
        return len(self.buffer) == self.sequence_length

    def get_window(self) -> np.ndarray:
        """
        Returns:
            (2, 5, 240)
        """
        if not self.is_full():
            raise ValueError("Buffer is not full yet")

        frames = list(self.buffer)

        # Remove height dimension (since it's always 1)
        frames = [f.squeeze(1) for f in frames]  # (2, 240)

        # Stack along time dimension
        x = np.stack(frames, axis=1)  # (2, 5, 240)

        return x


def print_window_summary(window):
    print("\nWindow shape:", window.shape)
    print("dtype:", window.dtype)

    print("range channel min/max:", window[0].min(), window[0].max())
    print("valid channel unique:", np.unique(window[1]))

if __name__ == "__main__":
    from pathlib import Path
    from test_preprocess_frame import load_raw_scan, preprocess_scan_to_frame

    folder = Path("/Users/karstenkempfe/Downloads/NewLidar-trainingModel/data/horses/dataset_1_Horsen")

    buffer = TemporalBuffer()

    for npz_path in sorted(folder.glob("*.npz"))[:10]:
        scan = load_raw_scan(npz_path)
        frame = preprocess_scan_to_frame(scan)

        buffer.add_frame(frame)
        print(f"Added frame: {npz_path.name}")

        if buffer.is_full():
            window = buffer.get_window()
            print_window_summary(window)
            break
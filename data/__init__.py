from torch.utils.data import DataLoader
from dataset import Synth90kDataset, synth90k_collate_fn

root = r"mnt/ramdisk/max/90kDICT32px"

dataset = Synth90kDataset(
    root_dir= root,
    mode="train",
)

batch = 8
loader = DataLoader(
    dataset,
    batch_size=batch,
    collate_fn = synth90k_collate_fn
)

for images, target, length in loader:
    print("img size:", images.shape)
    print("target:", target)
    print("length:", length)

    break

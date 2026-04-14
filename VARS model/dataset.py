from torch.utils.data import Dataset
from random import random
import torch
import random
from data_loader import label2vectormerge, clips2vectormerge
from torchvision.io.video import read_video


class MultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None, transform_model=None):

        if split != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.transform_model = transform_model
        self.num_views = num_views

        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)
        print(self.length)

        # Test-time augmentation knobs (used only in eval branch of __getitem__).
        # Set externally from a TTA driver script; defaults are no-op.
        self.tta_temporal_shift = 0
        self.tta_flip = False

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action,
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action,

    def _load_view(self, clip_path, temporal_shift=0, flip=False):
        """Load and process a single view clip."""
        video, _, _ = read_video(clip_path, output_format="THWC")

        window = self.end - self.start
        # Apply temporal shift (shift the window within available frames)
        start = self.start + temporal_shift
        end = start + window
        start = max(0, min(start, len(video) - window))
        end = start + window
        frames = video[start:end, :, :, :]

        # Pad short clips by repeating the last frame so all views align.
        if frames.shape[0] < window:
            pad_count = window - frames.shape[0]
            last = frames[-1:, :, :, :].repeat(pad_count, 1, 1, 1)
            frames = torch.cat((frames, last), dim=0)

        final_frames = None
        for j in range(len(frames)):
            if j % self.factor < 1:
                if final_frames is None:
                    final_frames = frames[j,:,:,:].unsqueeze(0)
                else:
                    final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0)

        final_frames = final_frames.permute(0, 3, 1, 2)

        if self.transform is not None:
            final_frames = self.transform(final_frames)

        final_frames = self.transform_model(final_frames)
        final_frames = final_frames.permute(1, 0, 2, 3)
        # final_frames: (C, T, H, W) — flip width dim for horizontal flip.
        if flip:
            final_frames = torch.flip(final_frames, dims=[-1])
        return final_frames

    def __getitem__(self, index):

        prev_views = []

        if self.split == 'Train':
            # Pick num_views random views from available clips (Run 1 behavior)
            # Random temporal shift (±10 frames ≈ 0.4s at 25fps) kept as cheap
            # aug — applied consistently across the sampled views.
            temporal_shift = random.randint(-10, 10)
            for num_view in range(len(self.clips[index])):
                index_view = random.randint(0, len(self.clips[index]) - 1)
                while index_view in prev_views:
                    index_view = random.randint(0, len(self.clips[index]) - 1)
                prev_views.append(index_view)
                if num_view == 0:
                    videos = self._load_view(self.clips[index][index_view], temporal_shift=temporal_shift).unsqueeze(0)
                else:
                    videos = torch.cat(
                        (videos, self._load_view(self.clips[index][index_view], temporal_shift=temporal_shift).unsqueeze(0)),
                        0,
                    )
                if num_view + 1 >= self.num_views:
                    break
        else:
            # Validation/Test/Chall: use all available views.
            # TTA knobs (tta_temporal_shift, tta_flip) default to no-op.
            shift = self.tta_temporal_shift
            flip = self.tta_flip
            for num_view in range(len(self.clips[index])):
                if num_view == 0:
                    videos = self._load_view(
                        self.clips[index][num_view],
                        temporal_shift=shift, flip=flip,
                    ).unsqueeze(0)
                else:
                    videos = torch.cat(
                        (videos, self._load_view(
                            self.clips[index][num_view],
                            temporal_shift=shift, flip=flip,
                        ).unsqueeze(0)),
                        0,
                    )

        videos = videos.permute(0, 2, 1, 3, 4)

        if self.split != 'Chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length

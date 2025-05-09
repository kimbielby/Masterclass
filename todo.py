# TODO: Segment person
#           Simulate green spill
#           - add a green halo around edges
#           - tint the skin, hair and clothes near the edges slightly green:
#               expand the mask of the person by a few pixels and blend green into these regions
#               add green to the shadows/reflections

# todo: OpenCV or PIL for basic blending
#         Use shader-like effects for more realism
#

# TODO:
#       1. Copy Train-Val-Test folders of LaPa images into dset folder
#       2. Run code on each of those folders to remove suspected greyscale images
#       3. Go through individually to remove remaining non-viable images
#           e.g. wearing green, sepia or greyscale-ish colour tone
#       4. Segment people - not just faces - and simulate green spill
#           - try on just a few images first
#       5. These will be saved in separate folder

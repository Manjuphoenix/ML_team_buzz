import Augmentor
p = Augmentor.Pipeline("dataset/Tomato___Target_Spot")
p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
p.shear(probability=0.5, max_shear_left=25, max_shear_right=25)
p.rotate(1, max_left_rotation=5, max_right_rotation=5)
p.flip_top_bottom(0.5)

# This completes the pipeline and creates 8000 images in a seperate folder called outputs in the provided path
p.sample(8000)

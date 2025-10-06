
def test_load_image():
  try:
    from A2_image_stitching import load_image

  except Exception as e:
    print(f"Failed test_load_image: {e}")


def test_extract_features():
  try:
    from A2_image_stitching import extract_features
  
  except Exception as e:
    print(f"Failed test_extract_features: {e}")

def test_match_features():
  try:
    from A2_image_stitching import match_features

  except Exception as e:
    print(f"Failed test_match_features: {e}")

def test_estimate_homography():
  try:
    from A2_image_stitching import estimate_homography

  except Exception as e:
    print(f"Failed test_estimate_homography: {e}")
    
def test_warp_image():
  try:
    from A2_image_stitching import warp_image

  except Exception as e:
    print(f"Failed test_warp_image: {e}")


def test_stitch_images():
  try:
    from A2_image_stitching import stitch_images

  except Exception as e:
    print(f"Failed test_stitch_images: {e}")


def main():
  test_load_image()
  test_extract_features()
  test_match_features()
  test_estimate_homography()
  test_warp_image()
  test_stitch_images()
  print("Done!")

if __name__ == "__main__":
  main()
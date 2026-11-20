from keras_cv.models import StableDiffusion
from PIL import Image

model = StableDiffusion(img_height=512, img_width=512, 
                        jit_compile=True)
img = model.text_to_image(
    prompt="A beautiful horse running through a field",
    batch_size=1,  # 生成多少張圖
    num_steps=25,  # 迭代次數 (可控制圖片品質)
    seed=123,      # 亂數種子, 相同提示可產生相同圖片
)
Image.fromarray(img[0]).save("horse.png")
print("儲存圖片: horse.png")

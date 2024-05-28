import fitz
from glob import glob
import os

pdfs = glob('./data/*.pdf')
fiugre_dir = './figures'

for pdf_path in pdfs:
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        image_list = page.get_images(full=True)


        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # 이미지 저장 경로
            image_filename = f"page{page_number + 1}_img{img_index + 1}.{image_ext}"
            image_path = os.path.join(fiugre_dir, image_filename)


            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

    print(f"{pdf_path}: 이미지 추출 완료")
    break
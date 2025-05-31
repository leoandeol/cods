import io
import logging
import math
import os

import torch
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from cods.od.data import MSCOCODataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = (
    "0"  # chose the GPU. If only one, then "0"
)

logging.getLogger().setLevel(logging.INFO)

# set [COCO_PATH] to the directory to your local copy of the COCO dataset
COCO_PATH = "/datasets/shared_datasets/coco/"

data = MSCOCODataset(root=COCO_PATH, split="val")
len(data)


def create_dataset_pdf(dataloader, output_filename="dataset_images.pdf"):
    # Set up the PDF canvas
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    # Register a default font
    pdfmetrics.registerFont(
        TTFont("Monospace", "monospace.medium.ttf"),
    )  #'arial.ttf'))

    # Calculate image size and positions
    image_width = width / 2 - 0.5 * inch
    image_height = (
        height / 2 - 0.75 * inch
    ) * 0.9  # Reduce image height to make room for title
    title_height = (height / 2 - 0.75 * inch) * 0.1  # Height for the title
    positions = [
        (0.25 * inch, height - 0.25 * inch - image_height - title_height),
        (
            width / 2 + 0.25 * inch,
            height - 0.25 * inch - image_height - title_height,
        ),
        (0.25 * inch, 0.25 * inch + title_height),
        (width / 2 + 0.25 * inch, 0.25 * inch + title_height),
    ]

    image_count = 0
    for batch in dataloader:
        image_paths, image_sizes, images, ground_truth = batch

        for img, path in zip(images, image_paths):
            # Calculate position for this image
            pos = positions[image_count % 4]

            # Convert JpegImageFile to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="JPEG")
            img_byte_arr = img_byte_arr.getvalue()

            # Create an ImageReader object
            img_reader = ImageReader(io.BytesIO(img_byte_arr))

            # Draw image
            c.drawImage(
                img_reader,
                pos[0],
                pos[1],
                width=image_width,
                height=image_height,
            )

            # Draw title (image path)
            c.setFont("Monospace", 8)
            title = os.path.basename(
                path,
            )  # Use only the filename, not the full path
            title_width = c.stringWidth(title, "Monospace", 8)
            if title_width > image_width:
                # If title is too long, truncate it
                while title_width > image_width and len(title) > 3:
                    title = (
                        title[:-4] + "..."
                    )  # Remove 3 characters and add ellipsis
                    title_width = c.stringWidth(title, "Monospace", 8)
            c.drawString(
                pos[0] + (image_width - title_width) / 2,
                pos[1] - title_height / 2,
                title,
            )

            image_count += 1

            # Start a new page if we've filled this one
            if image_count % 4 == 0:
                c.showPage()

    # Save the PDF
    c.save()

    print(
        f"PDF created with {image_count} images on {math.ceil(image_count / 4)} pages.",
    )


# Use the function
dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=64,
    shuffle=False,
    collate_fn=data._collate_fn,
)
create_dataset_pdf(dataloader)

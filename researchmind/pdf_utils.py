from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import re


def create_pdf(report_text, topic):

    safe_topic = re.sub(
        r'[^a-zA-Z0-9_-]',
        '_',
        topic
    )

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{safe_topic}.pdf")

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            f"<b>Research Report:</b> {topic}",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 12)
    )

    content.append(
        Paragraph(
            report_text.replace("\n", "<br/>"),
            styles["BodyText"]
        )
    )

    doc.build(content)

    return filename
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def create_pdf(report_text, topic):

    filename = f"{topic.replace(' ', '_')}.pdf"

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            f"<b>Research Report:</b> {topic}",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            report_text.replace("\n", "<br/>"),
            styles["BodyText"]
        )
    )

    doc.build(content)

    return filename
from fpdf import FPDF

class PDF(FPDF):
    def __init__(self, pdf_title, orientation='P', unit='mm', format='A4', font_cache_dir=None) -> None:
        super().__init__(orientation, unit, format)
        
        self.pdf_title = pdf_title
        
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, self.pdf_title, align="C", ln=True)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Página {self.page_no()}", align="C")

# pdf = PDF(pdf_title='Async Report')
# pdf.add_page()
# pdf.set_font("Arial", size=12)
# pdf.cell(200, 10, txt="Olá, Mundo!", ln=True)
# pdf.cell(200, 10, txt="Olá, Mundo!", ln=True)
# pdf.cell(200, 10, txt="Olá, Mundo!", ln=True)
# pdf.cell(200, 10, txt="Olá, Mundo!", ln=True)
# pdf.cell(200, 10, txt="Olá, Mundo!", ln=True)

# pdf.add_page()
# pdf.set_font("Arial", size=12)
# pdf.cell(200, 10, txt="Olá, Feira!", ln=True)
# pdf.cell(200, 10, txt="Olá, Feira!", ln=True)
# pdf.cell(200, 10, txt="Olá, Feira!", ln=True)
# pdf.cell(200, 10, txt="Olá, Feira!", ln=True)
# pdf.cell(200, 10, txt="Olá, Feira!", ln=True)

# pdf.output("meu_documento.pdf")

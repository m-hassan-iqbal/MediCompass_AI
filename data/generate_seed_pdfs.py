"""
Generate high-fidelity seed PDFs for MediCompass AI demo knowledge base.
Creates valid multi-page PDF files readable by pypdf.
"""
import os
import pypdf

def create_multipage_pdf(filename: str, pages_text: list[str]) -> None:
    """Build a valid PDF 1.4 file containing multiple pages with extracted text."""
    num_pages = len(pages_text)
    font_obj_id = 3 + (num_pages * 2)
    page_obj_ids = [3 + (i * 2) for i in range(num_pages)]
    content_obj_ids = [4 + (i * 2) for i in range(num_pages)]
    
    catalog = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    kids = " ".join([f"{pid} 0 R" for pid in page_obj_ids])
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj\n"
    body = [catalog, pages_obj]
    
    for i, text in enumerate(pages_text):
        pid = page_obj_ids[i]
        cid = content_obj_ids[i]
        lines = text.strip().split("\n")
        stream_cmds = ["BT", "/F1 11 Tf", "50 740 Td", "14 TL"]
        for idx, line in enumerate(lines):
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if idx == 0:
                stream_cmds.append(f"({safe_line}) Tj")
            else:
                stream_cmds.append(f"T* ({safe_line}) Tj")
        stream_cmds.append("ET")
        stream_str = "\n".join(stream_cmds)
        stream_bytes = stream_str.encode("latin-1", "replace")
        
        page_entry = (
            f"{pid} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 {font_obj_id} 0 R >> >> >>\n"
            f"endobj\n"
        )
        content_entry = (
            f"{cid} 0 obj\n"
            f"<< /Length {len(stream_bytes)} >>\n"
            f"stream\n{stream_str}\nendstream\n"
            f"endobj\n"
        )
        body.append(page_entry)
        body.append(content_entry)
        
    font_entry = (
        f"{font_obj_id} 0 obj\n"
        f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
        f"endobj\n"
    )
    body.append(font_entry)
    
    pdf_content = "%PDF-1.4\n"
    offsets = [0]
    curr_offset = len(pdf_content.encode("latin-1"))
    
    for item in body:
        offsets.append(curr_offset)
        curr_offset += len(item.encode("latin-1"))
        pdf_content += item
        
    xref_offset = curr_offset
    total_objs = font_obj_id + 1
    xref = f"xref\n0 {total_objs}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"
        
    trailer = (
        f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    )
    pdf_content += xref + trailer
    
    with open(filename, "wb") as f:
        f.write(pdf_content.encode("latin-1"))

def main():
    target_dir = os.path.join(os.path.dirname(__file__), "sources")
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. Punjab Biology Ch 11
    punjab_pages = [
        """PUNJAB TEXTBOOK BOARD - BIOLOGY INTERMEDIATE PART-I
CHAPTER 11: ENZYMES AND METABOLISM (Page 1)
Enzymes are biological catalysts that speed up chemical reactions without being consumed in the process.
Every enzyme contains an active site with binding and catalytic sites.
Enzymes lower the activation energy of biological reactions, accelerating their velocity.
Apoenzyme is the protein part of enzyme requiring a non-protein cofactor or coenzyme to form a holoenzyme.
Specific enzyme substrate interactions govern biological metabolism in living organisms.""",

        """PUNJAB TEXTBOOK BOARD - BIOLOGY INTERMEDIATE PART-I
CHAPTER 11: ENZYMES (Page 2) - MECHANISM OF ENZYMATIC CATALYSIS
Emil Fischer proposed the Lock and Key Model in 1894, stating that enzyme active site is rigid and specific.
Daniel Koshland proposed the Induced Fit Model in 1958, proposing conformational change upon substrate binding.
Factors affecting enzyme catalysis:
1. Temperature: Optimal human enzyme temperature is 37 C. Denaturation occurs at extreme temperatures.
2. pH: Each enzyme has an optimum pH (e.g. Pepsin pH 2.0, Trypsin pH 8.0).
3. Substrate concentration: Reaction rate increases until active sites become saturated (Vmax reached).""",

        """PUNJAB TEXTBOOK BOARD - BIOLOGY INTERMEDIATE PART-I
CHAPTER 11: ENZYMES (Page 3) - ENZYME INHIBITION AND KINETICS
An inhibitor is a substance that reduces or ceases enzyme catalytic activity.
Competitive Inhibition:
A competitive inhibitor possesses structural resemblance to the natural substrate.
It competes directly for the active site of the enzyme.
When the competitive inhibitor binds, it prevents substrate binding, forming an enzyme-inhibitor complex.
Crucial rule: Competitive inhibition can be completely overcome by increasing the substrate concentration.
Because substrate competition is surmountable, the maximum velocity (Vmax) remains unaltered.
However, more substrate is required to achieve half-maximum velocity; therefore, apparent Km increases.
Classic example: Malonate inhibits succinate dehydrogenase by competing with succinate.""",

        """PUNJAB TEXTBOOK BOARD - BIOLOGY INTERMEDIATE PART-I
CHAPTER 11: ENZYMES (Page 4) - NON-COMPETITIVE AND IRREVERSIBLE INHIBITION
Non-competitive inhibitors do not compete for the active site.
They bind to an allosteric site (a distinct regulatory site away from the catalytic center).
Binding changes the three-dimensional globular conformation of the enzyme, rendering the active site inactive.
Adding excess substrate CANNOT overcome non-competitive inhibition.
Consequently, maximum velocity (Vmax) decreases significantly.
Because the affinity of the uninhibited enzyme molecules for substrate remains unchanged, Km remains constant.
Examples: Heavy metals like lead and mercury, and cyanide poisoning cytochrome oxidase."""
    ]
    create_multipage_pdf(os.path.join(target_dir, "Punjab_Biology_Enzymes_Ch11.pdf"), punjab_pages)
    
    # 2. Federal Biology Ch 3
    federal_pages = [
        """FEDERAL BOARD / NATIONAL BOOK FOUNDATION - BIOLOGY CLASS XI
CHAPTER 3: ENZYMES AND BIOENERGETICS (Page 1)
Enzymes operate as biocatalysts by stabilizing transition state complexes.
The active site is composed of catalytic residues that directly participate in bond breaking and formation.
Allosteric regulation allows cellular feedback loops where end-products act as feedback inhibitors.
Cofactors can be inorganic metal ions (like Mg2+, Zn2+, Fe2+) or organic coenzymes like NAD+ and FAD.""",

        """FEDERAL BOARD / NATIONAL BOOK FOUNDATION - BIOLOGY CLASS XI
CHAPTER 3: ENZYME KINETICS & MICHAELIS-MENTEN CONSTANT (Page 2)
The Michaelis constant (Km) represents the substrate concentration at which reaction rate is half of Vmax (1/2 Vmax).
Km reflects the enzyme affinity for its substrate:
A low Km denotes high substrate affinity, meaning less substrate is required to saturate active sites.
A high Km denotes lower substrate affinity, requiring higher substrate concentrations.
In competitive inhibition:
The inhibitor occupies the active site. Apparent affinity is reduced, leading to an increased Km.
At infinite substrate concentration, substrate outcompetes inhibitor, so Vmax is unchanged.
In non-competitive inhibition:
Inhibitor alters enzyme turnover number (kcat), causing Vmax to decrease, while Km remains unchanged.""",

        """FEDERAL BOARD / NATIONAL BOOK FOUNDATION - BIOLOGY CLASS XI
CHAPTER 3: PHARMACOLOGY AND CLINICAL RELEVANCE (Page 3)
Medical applications of enzyme inhibitors:
1. Sulfa drugs (Sulfanilamide) serve as competitive inhibitors of dihydropteroate synthetase in bacteria.
Sulfanilamide resembles para-aminobenzoic acid (PABA), starving bacteria of essential folic acid.
2. Penicillin is an irreversible inhibitor of transpeptidase, preventing bacterial cell wall cross-linking.
3. Organophosphate nerve agents irreversibly phosphorylate acetylcholinesterase, causing cholinergic crisis."""
    ]
    create_multipage_pdf(os.path.join(target_dir, "Federal_Biology_Enzymes_Ch3.pdf"), federal_pages)
    
    # 3. PMDC MDCAT & NUMS Syllabus
    syllabus_pages = [
        """PAKISTAN MEDICAL AND DENTAL COUNCIL (PMDC)
OFFICIAL MDCAT & NUMS CURRICULUM - BIOLOGY SECTION (Page 1)
Section 2: Biological Molecules, Enzymes, and Cellular Kinetics
Core Learning Outcomes for MDCAT and NUMS candidates:
Outcome 2.1: Outline the chemical nature and characteristics of globular enzyme proteins.
Outcome 2.2: Differentiate clearly between the Lock and Key hypothesis and the Induced Fit Model.
Outcome 2.3: Analyze factors influencing rate of enzyme catalyzed reactions (temperature, pH, substrate).
Outcome 2.4: Distinguish between competitive and non-competitive inhibitors in terms of:
  - Binding site (active site vs allosteric site)
  - Structural similarity to substrate
  - Effect of increasing substrate concentration on overcoming inhibition
  - Kinetic changes: effect on Km and effect on Vmax
Outcome 2.5: Discuss medical and physiological importance of inhibitors (sulfa drugs, poisons).
Exam Weightage: High yield core concept with frequent multi-year conceptual items."""
    ]
    create_multipage_pdf(os.path.join(target_dir, "PMDC_MDCAT_NUMS_Syllabus_Biology.pdf"), syllabus_pages)
    print("Seed PDFs generated successfully in data/sources/")

if __name__ == "__main__":
    main()

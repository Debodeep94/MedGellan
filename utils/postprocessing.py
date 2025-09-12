def extract_icd10_codes(record: str) -> list:
    import re
    icd10_pattern = r"[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?"
    codes = re.findall(icd10_pattern, record)
    return codes if codes else ["NA"]


def get_icd10_chapter(code):
    code = code.upper().strip()
    chapter_ranges = [
        ('A00', 'B99', 'Infectious diseases'),
        ('C00', 'D49', 'Neoplasms'),
        ('D50', 'D89', 'Blood diseases'),
        ('E00', 'E89', 'Endocrine diseases'),
        ('F01', 'F99', 'Mental disorders'),
        ('G00', 'G99', 'Nervous system'),
        ('H00', 'H59', 'Eye diseases'),
        ('H60', 'H95', 'Ear diseases'),
        ('I00', 'I99', 'Circulatory system'),
        ('J00', 'J99', 'Respiratory system'),
        ('K00', 'K95', 'Digestive system'),
        ('L00', 'L99', 'Skin diseases'),
        ('M00', 'M99', 'Musculoskeletal system'),
        ('N00', 'N99', 'Genitourinary system'),
        ('O00', 'O99', 'Pregnancy & childbirth'),
        ('P00', 'P96', 'Perinatal conditions'),
        ('Q00', 'Q99', 'Congenital malformations'),
        ('R00', 'R99', 'Symptoms/signs'),
        ('S00', 'T88', 'Injuries & poisonings'),
        ('V00', 'Y99', 'External causes'),
        ('Z00', 'Z99', 'Factors influencing health'),
    ]

    if len(code) < 3:
        return 'Unknown'
    code_prefix = code[:3]
    for start, end, chapter in chapter_ranges:
        if start <= code_prefix <= end:
            return chapter
    return 'Unknown'

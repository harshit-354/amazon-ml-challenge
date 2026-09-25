"""
Unit tests for Normalization Pipeline
"""

import unittest
import pandas as pd
from src.normalization import (
    normalize_name,
    normalize_address,
    normalize_record,
    normalize_dataframe,
)


class TestNormalization(unittest.TestCase):

    def test_problem_statement_examples(self):
        """Verify the exact examples provided in the problem statement."""
        raw_name = "Shree Krishna Medical Store Pvt. Ltd."
        expected_name = "shree krishna medical store"
        self.assertEqual(normalize_name(raw_name), expected_name)

        raw_addr = "12, M.G. Road, Anand"
        expected_addr = "12 mg road anand"
        self.assertEqual(normalize_address(raw_addr), expected_addr)

    def test_legal_suffixes_removal(self):
        """Verify legal suffix stripping across multiple jurisdictions and styles."""
        # US / UK suffixes
        self.assertEqual(normalize_name("Apple Inc."), "apple")
        self.assertEqual(normalize_name("Obsidian, LLC"), "obsidian")
        self.assertEqual(normalize_name("Obsidian,-LLC"), "obsidian")
        self.assertEqual(normalize_name("Obsidian, [[LLC]]"), "obsidian")
        self.assertEqual(normalize_name("[Corp] Dick Regional Armada"), "dick regional armada")
        self.assertEqual(normalize_name("LLC Crystal Staffing Solutions"), "crystal staffing solutions")
        self.assertEqual(normalize_name("CRYSTAL STAFFING SOLUTIONS-L.L.C."), "crystal staffing solutions")
        
        # Indian suffixes
        self.assertEqual(normalize_name("Consulting Nyasa Nursing Private Limited"), "consulting nyasa nursing")
        self.assertEqual(normalize_name("Consulting Nyasa Nursing Private Ltd"), "consulting nyasa nursing")
        self.assertEqual(normalize_name("CONSULTING NYASA NURSING PRIVATE LIMITED"), "consulting nyasa nursing")
        self.assertEqual(normalize_name("Consulting Nyasa Nursing Private"), "consulting nyasa nursing")
        self.assertEqual(normalize_name("Raj Investments LLP"), "raj investments")

        # French suffixes
        self.assertEqual(normalize_name("Thermal & Fils SASU"), "thermal and fils")
        self.assertEqual(normalize_name("Grillons Lycee SAS"), "grillons lycee")
        self.assertEqual(normalize_name("ZNB Club SARL"), "znb club")
        self.assertEqual(normalize_name("Elephant Centre EURL"), "elephant centre")
        self.assertEqual(normalize_name("Collège Jean SA"), "college jean")

    def test_symbols_and_abbreviations(self):
        """Verify symbols (&, +, @) and corporate abbreviation expansions."""
        self.assertEqual(normalize_name("Hendricks & Flowers"), "hendricks and flowers")
        self.assertEqual(normalize_name("Chordia + Partners"), "chordia and partners")
        self.assertEqual(normalize_name("<< Team Ecole"), "team ecole")
        self.assertEqual(normalize_name("Apex Mfg & Tech Solns"), "apex manufacturing and technology solutions")

    def test_indic_names_transliteration(self):
        """Verify Indic scripts transliteration and suffix stripping."""
        self.assertEqual(normalize_name("एसएस फूड प्राइवेट लिमिटेड"), "ss food")
        self.assertEqual(normalize_name("Ss Food Private Limited"), "ss food")
        self.assertEqual(normalize_name("Raj Investments எல்எல்பி"), "raj investments")

    def test_dba_handling(self):
        """Verify DBA / trade name extraction."""
        self.assertEqual(normalize_name("Korbrixx D.B.A. Obsidian, LLC"), "obsidian")
        self.assertEqual(normalize_name("John Doe d/b/a Acme Tools"), "acme tools")

    def test_address_street_abbreviations(self):
        """Verify road, street, avenue, boulevard expansions."""
        self.assertEqual(normalize_address("85 Wayne Ave"), "85 wayne avenue")
        self.assertEqual(normalize_address("3315 Fremont St"), "3315 fremont street")
        self.assertEqual(normalize_address("3315 Fremont Saint"), "3315 fremont street")
        self.assertEqual(normalize_address("154 BD du President Wilson"), "154 boulevard du president wilson")
        self.assertEqual(normalize_address("16 R. DES GRILLONS"), "16 rue des grillons")

    def test_address_state_expansion(self):
        """Verify state name expansions."""
        self.assertEqual(normalize_address("Peoria, IL", country="US"), "peoria illinois")
        self.assertEqual(normalize_address("Ticonderoga, NY", country="US"), "ticonderoga new york")
        self.assertEqual(normalize_address("Chennai, TN", country="India"), "chennai tamil nadu")
        self.assertEqual(normalize_address("Mumbai, MH", country="India"), "mumbai maharashtra")
        self.assertEqual(normalize_address("Ghaziabad, UP", country="India"), "ghaziabad uttar pradesh")

    def test_indic_script_addresses(self):
        """Verify Indic script state translation."""
        self.assertEqual(
            normalize_address("BHANDUP WEST, MUMBAI, महाराष्ट्र"),
            "bhandup west mumbai maharashtra"
        )
        self.assertEqual(
            normalize_address("CHENNAI, தமிழ்நாடு"),
            "chennai tamil nadu"
        )
        self.assertEqual(
            normalize_address("GHAZIABAD, उत्तर प्रदेश"),
            "ghaziabad uttar pradesh"
        )

    def test_ordinals_and_number_formatting(self):
        """Verify ordinals (1st, 2nd, 45th/45nd) and leading zero cleanup."""
        self.assertEqual(normalize_address("630 45th Terrace"), "630 45 terrace")
        self.assertEqual(normalize_address("630 45nd Terrace"), "630 45 terrace")
        self.assertEqual(normalize_address("AF-0684, Nandgram"), "af 684 nandgram")
        self.assertEqual(normalize_address("022 Allée Des Grillons"), "22 allee des grillons")

    def test_null_and_empty_inputs(self):
        """Verify graceful handling of null, None, nan, and empty strings."""
        self.assertEqual(normalize_name(""), "")
        self.assertEqual(normalize_name(None), "")
        self.assertEqual(normalize_name("null"), "")
        self.assertEqual(normalize_name("<NULL>"), "")
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address(None), "")
        self.assertEqual(normalize_address("<NULL>"), "")

    def test_dataframe_normalization(self):
        """Verify batch DataFrame normalization."""
        data = {
            "entity_id": ["S1-001", "S1-002"],
            "business_name": ["Shree Krishna Medical Store Pvt. Ltd.", "Vision Partners Corp"],
            "business_address": ["12, M.G. Road, Anand", "1064 Newton Rd, Unit 11, IA"],
            "country": ["India", "US"],
        }
        df = pd.DataFrame(data)
        norm_df = normalize_dataframe(df)

        self.assertIn("norm_name", norm_df.columns)
        self.assertIn("norm_address", norm_df.columns)
        self.assertEqual(norm_df["norm_name"].iloc[0], "shree krishna medical store")
        self.assertEqual(norm_df["norm_address"].iloc[0], "12 mg road anand")
        self.assertEqual(norm_df["norm_name"].iloc[1], "vision partners")
        self.assertEqual(norm_df["norm_address"].iloc[1], "1064 newton road unit 11 iowa")


if __name__ == "__main__":
    unittest.main()

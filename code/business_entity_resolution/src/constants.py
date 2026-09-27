"""
ML Challenge: Business Entity Resolution
Constants and Reference Dictionaries for Normalization
"""

# Indic scripts state mapping -> canonical English names
INDIC_STATES = {
    'महाराष्ट्र': 'maharashtra',
    'उत्तर प्रदेश': 'uttar pradesh',
    'उत्तरप्रदेश': 'uttar pradesh',
    'தமிழ்நாடு': 'tamil nadu',
    'கர்ನಾಟಕ': 'karnataka',
    'हरियाणा': 'haryana',
    'राजस्थान': 'rajasthan',
    'दिल्ली': 'delhi',
    'गुजरात': 'gujarat',
    'पश्चिम बंगाल': 'west bengal',
    'केरल': 'kerala',
    'पंजाब': 'punjab',
    'आंध्र प्रदेश': 'andhra pradesh',
    'आन्ध्र प्रदेश': 'andhra pradesh',
    'मध्य प्रदेश': 'madhya pradesh',
    'बिहार': 'bihar',
    'ओडिशा': 'odisha',
    'उड़ीसा': 'odisha',
    'तेलंगाना': 'telangana',
    'गोवा': 'goa',
    'असम': 'assam',
    'झारखंड': 'jharkhand',
    'उत्तराखंड': 'uttarakhand',
    'छत्तीसगढ़': 'chhattisgarh',
    'हिमाचल प्रदेश': 'himachal pradesh',
    'जम्मू और कश्मीर': 'jammu and kashmir',
    'पुडुचेरी': 'puducherry',
    'चंडीगढ़': 'chandigarh',
    'उत्तरांचल': 'uttarakhand',
}

# Indic legal suffixes in native scripts
INDIC_LEGAL_SUFFIXES = [
    r'प्राइवेट\s+लिमिटेड',
    r'प्राइवेट\s+लि\b\.?',
    r'प्रा\.\s*लि\b\.?',
    r'प्रा\.\s*लिमिटेड',
    r'प्रा\s*लि',
    r'लिमिटेड',
    r'कंपनी',
    r'एलएलपी',
    r'பிரைவேட்\s+லிமிடெட்',
    r'லிமிடெட்',
    r'எல்எல்பி',
    r'ಪ್ರೈವೇಟ್\s+ಲಿಮಿಟೆಡ್',
    r'ಲಿಮಿಟೆಡ್',
]

# US State abbreviations (lowercase) -> full state name
US_STATES = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas', 'ca': 'california',
    'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware', 'fl': 'florida', 'ga': 'georgia',
    'hi': 'hawaii', 'id': 'idaho', 'il': 'illinois', 'in': 'indiana', 'ia': 'iowa',
    'ks': 'kansas', 'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada', 'nh': 'new hampshire',
    'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york', 'nc': 'north carolina',
    'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma', 'or': 'oregon', 'pa': 'pennsylvania',
    'ri': 'rhode island', 'sc': 'south carolina', 'sd': 'south dakota', 'tn': 'tennessee',
    'tx': 'texas', 'ut': 'utah', 'vt': 'vermont', 'va': 'virginia', 'wa': 'washington',
    'wv': 'west virginia', 'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
    'pr': 'puerto rico',
}

# Indian State abbreviations (lowercase) -> full state name
INDIA_STATES = {
    'mh': 'maharashtra', 'dl': 'delhi', 'ka': 'karnataka', 'tn': 'tamil nadu',
    'up': 'uttar pradesh', 'gj': 'gujarat', 'wb': 'west bengal', 'tg': 'telangana',
    'ts': 'telangana', 'ap': 'andhra pradesh', 'rj': 'rajasthan', 'mp': 'madhya pradesh',
    'kl': 'kerala', 'pb': 'punjab', 'hr': 'haryana', 'od': 'odisha', 'or': 'odisha',
    'br': 'bihar', 'as': 'assam', 'jh': 'jharkhand', 'uk': 'uttarakhand', 'ua': 'uttarakhand',
    'ch': 'chandigarh', 'ga': 'goa', 'hp': 'himachal pradesh', 'cg': 'chhattisgarh',
    'sk': 'sikkim', 'tr': 'tripura', 'mn': 'manipur', 'ml': 'meghalaya', 'mz': 'mizoram',
    'nl': 'nagaland', 'ar': 'arunachal pradesh', 'jk': 'jammu and kashmir', 'la': 'ladakh',
    'py': 'puducherry', 'an': 'andaman and nicobar', 'dn': 'dadra and nagar haveli',
    'dd': 'daman and diu',
}

# Address abbreviations and expansions (US, India, France)
ADDRESS_ABBR = {
    # Street types (US / UK / General)
    'rd': 'road', 'st': 'street', 'saint': 'street', 'ave': 'avenue', 'av': 'avenue',
    'blvd': 'boulevard', 'bd': 'boulevard', 'bld': 'boulevard', 'bvd': 'boulevard',
    'ln': 'lane', 'dr': 'drive', 'ct': 'court', 'pl': 'place', 'sq': 'square',
    'hwy': 'highway', 'pkwy': 'parkway', 'cir': 'circle', 'rte': 'route', 'rt': 'route',
    'terr': 'terrace', 'ter': 'terrace', 'expy': 'expressway', 'fwy': 'freeway',
    'aly': 'alley', 'way': 'way', 'trl': 'trail', 'row': 'row', 'loop': 'loop',
    
    # French street words & prefixes
    'r': 'rue', 'all': 'allee', 'imp': 'impasse', 'chem': 'chemin', 'ch': 'chemin',
    'crs': 'cours', 'fg': 'faubourg', 'qu': 'quai', 'pass': 'passage',
    
    # Secondary units & buildings
    'apt': 'apartment', 'apts': 'apartment', 'ste': 'suite', 'fl': 'floor', 'flr': 'floor',
    'bldg': 'building', 'rm': 'room', 'dept': 'department', 'sec': 'sector', 'sect': 'sector',
    'twp': 'township', 'townshiip': 'township',
    
    # Indian locality terms
    'clg': 'colony', 'cln': 'colony', 'col': 'colony', 'soc': 'society', 'socty': 'society',
    'ngr': 'nagar', 'mrg': 'marg', 'chwk': 'chowk', 'bzr': 'bazaar',
    'ind': 'industrial', 'indl': 'industrial', 'est': 'estate', 'dist': 'district',
    'pk': 'park', 'plz': 'plaza', 'ctr': 'center', 'cntr': 'center', 'mkt': 'market',
    'opp': 'opposite', 'oppo': 'opposite', 'nr': 'near', 'bh': 'behind',
    'ext': 'extension', 'extn': 'extension', 'ph': 'phase',
    
    # Cardinal directions
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
}

# Legal suffixes to strip from business names (longest first)
LEGAL_SUFFIXES = [
    'private limited', 'pvt limited', 'private ltd', 'p limited', 'p ltd', 'pvt ltd', 'pvtltd',
    'limited liability company', 'limited liability partnership', 'limited partnership',
    'praaivett limittedd', 'praaivett limitted', 'limittedd', 'elelpii', 'elelpi', 'knpnii',
    'llc', 'llp', 'pllc', 'lp', 'pc',
    'corporation', 'corp', 'incorporated', 'inc',
    'company', 'co',
    'limited', 'ltd', 'private', 'pvt',
    'sarl', 'sas', 'sasu', 'sa', 'eurl', 'sci', 'snc', 'gie', 'sca', 'scs', 'selarl',
    'gmbh', 'ag', 'bv', 'nv', 'plc',
]

# Business name common abbreviations
NAME_ABBR = {
    'mfg': 'manufacturing', 'manuf': 'manufacturing',
    'tech': 'technology', 'technol': 'technology',
    'intl': 'international', 'intl.': 'international',
    'svc': 'services', 'svcs': 'services', 'serv': 'services',
    'dept': 'department', 'univ': 'university', 'hosp': 'hospital',
    'assoc': 'associates', 'assn': 'associates',
    'med': 'medical', 'pharma': 'pharmaceuticals',
    'comm': 'communications', 'comms': 'communications',
    'grp': 'group', 'soln': 'solutions', 'solns': 'solutions',
    'eng': 'engineering', 'engg': 'engineering',
    'dist': 'distributor', 'distrib': 'distributor',
    'sys': 'systems', 'syst': 'systems',
    'natl': 'national', 'ind': 'industries', 'inds': 'industries',
    'ent': 'enterprises', 'enterp': 'enterprises',
    'prod': 'products', 'prods': 'products',
    'dev': 'development', 'gen': 'general',
    'chem': 'chemicals', 'chems': 'chemicals',
    'cons': 'consulting', 'lab': 'laboratories', 'labs': 'laboratories',
    'mgmt': 'management', 'auto': 'automotive',
}

# Phonetic / transliteration cleanup for Indic words
INDIC_PHONETIC_REPLACEMENTS = [
    (r'\b(?:eses|es\s+es)\b', 'ss'),
    (r'\bphuudd\b', 'food'),
    (r'\belelpii?\b', 'llp'),
]

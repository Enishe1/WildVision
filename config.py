# 100 species from Bosnia and Herzegovina

TARGET_SPECIES = [
    # Mammals (25)
    "Lynx lynx", "Ursus arctos", "Canis lupus", "Capreolus capreolus", "Sus scrofa",
    "Martes martes", "Vulpes vulpes", "Lutra lutra", "Rupicapra rupicapra", "Myotis myotis",
    "Sciurus vulgaris", "Lepus europaeus", "Felis silvestris", "Meles meles", "Neomys fodiens",
    "Cervus elaphus", "Dama dama", "Mustela nivalis", "Glis glis", "Talpa europaea",
    "Miniopterus schreibersii", "Rhinolophus ferrumequinum", "Nyctalus noctula", "Erinaceus europaeus", "Microtus arvalis",
    # Birds (35)
    "Aquila chrysaetos", "Falco peregrinus", "Bubo bubo", "Ciconia ciconia", "Picus viridis",
    "Lanius collurio", "Sitta europaea", "Turdus merula", "Emberiza citrinella", "Alcedo atthis",
    "Milvus migrans", "Corvus corax", "Parus major", "Motacilla alba", "Hirundo rustica",
    "Phylloscopus collybita", "Luscinia megarhynchos", "Accipiter nisus", "Strix aluco", "Apus apus",
    "Columba palumbus", "Anas platyrhynchos", "Ardea cinerea", "Upupa epops", "Pyrrhocorax pyrrhocorax",
    "Falco tinnunculus", "Sylvia atricapilla", "Regulus regulus", "Carduelis carduelis", "Streptopelia decaocto",
    "Certhia brachydactyla", "Jynx torquilla", "Dryocopus martius", "Turdus philomelos", "Passer domesticus",
    # Reptiles & Amphibians (15)
    "Vipera berus", "Zamenis longissimus", "Natrix natrix", "Lacerta viridis", "Podarcis muralis",
    "Triturus carnifex", "Bombina variegata", "Rana temporaria", "Salamandra salamandra", "Bufo bufo",
    "Anguis fragilis", "Emys orbicularis", "Hyla arborea", "Pelophylax kl. esculentus", "Coronella austriaca",
    # Fish (10)
    "Salmo trutta", "Hucho hucho", "Barbus balcanicus", "Squalius cephalus", "Alburnus alburnus",
    "Esox lucius", "Silurus glanis", "Thymallus thymallus", "Cottus gobio", "Phoxinus phoxinus",
    # Insects (15)
    "Lucanus cervus", "Papilio machaon", "Apis mellifera", "Calopteryx splendens", "Carabus intricatus",
    "Rosalia alpina", "Morimus funereus", "Lycaena dispar", "Vespa crabro", "Polyommatus icarus",
    "Odonata", "Cetonia aurata", "Bombus terrestris", "Limenitis reducta", "Mantis religiosa"
]

# Model configuration
MODEL_CONFIG = {
    "image_size": (300, 300),
    "batch_size": 32,
    "epochs": 30,
    "max_images_per_species": 200,
    "test_size": 0.2,
    "base_model": "EfficientNetB3"
}

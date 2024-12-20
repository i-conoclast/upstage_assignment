E1_MARKER = "[E1]"
E1_MARKER_CLOSE = "[/E1]"
E2_MARKER = "[E2]"
E2_MARKER_CLOSE = "[/E2]"
E1_MARKER_PREFIX = "[E1"
E1_MARKER_CLOSE_PREFIX = "[/E1"
E2_MARKER_PREFIX = "[E2"
E2_MARKER_CLOSE_PREFIX = "[/E2"

LABELS_LIST = [
 'no_relation',
 'org:top_members/employees',
 'org:members',
 'org:product',
 'per:title',
 'org:alternate_names',
 'per:employee_of',
 'org:place_of_headquarters',
 'per:product',
 'org:number_of_employees/members',
 'per:children',
 'per:place_of_residence',
 'per:alternate_names',
 'per:other_family',
 'per:colleagues',
 'per:origin',
 'per:siblings',
 'per:spouse',
 'org:founded',
 'org:political/religious_affiliation',
 'org:member_of',
 'per:parents',
 'org:dissolved',
 'per:schools_attended',
 'per:date_of_death',
 'per:date_of_birth',
 'per:place_of_birth',
 'per:place_of_death',
 'org:founded_by',
 'per:religion'
]

MODEL_NAME_OR_PATH = "klue/roberta-large"
NUM_LABELS = len(LABELS_LIST)
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
MAX_LENGTH = 128
LABEL_SMOOTHING = 0.09
ALPHA = 1.0
GAMMA = 2.0

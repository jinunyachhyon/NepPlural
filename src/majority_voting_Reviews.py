import pandas as pd
import re

# Load the file needing human review fix
df = pd.read_csv('IDS_migration_old_podcast_Annotated.csv')

def auto_fix_review(row):
    comm = str(row['comment']).lower()
    intent = str(row['intent'])
    driver = str(row['primary_driver'])
    orient = str(row['value_orientation'])
    affect = str(row['affect'])
    
    # Context flags
    is_meta = any(k in comm for k in ['like your video', 'nice video', 'good podcast', 'great podcast', 'podcast is good', 'educational and entertaining', 'types of communication', 'repost', 'very nice', 'face reveal', 'best content', 'quality content'])
    is_lang = any(k in comm for k in ['speak full english', 'purai english', 'nepali mai bolnu', 'language', 'bhaso', 'bhasa'])
    is_fact_corr = any(k in comm for k in ['3 million', '30 million', '3million', '30million', '3 crore', '3crore', 'wrong data', 'data mistake', 'data is wrong'])
    is_neta_sarkar = any(k in comm for k in ['neta', 'sarkar', 'corruption', 'bhrastachar', 'pm', 'oli', 'deuba', 'prachanda', 'politician', 'government'])
    is_family = any(k in comm for k in ['family', 'baba', 'mami', 'pariwar', 'aama', 'buwa', 'parents', 'xora', 'xori', 'pariwar'])
    is_job_money = any(k in comm for k in ['job', 'unemployment', 'salary', 'paisa', 'money', 'earn', 'income', 'tu ', 'university', 'bachelor', 'it sector', 'freelance'])

    # Fix Intent
    if intent == 'Needs Human Review':
        if any(k in comm for k in ['ma pani janchu', 'leaving', 'process gari', 'flight', 'apply gare', 'bidesh nai last option', 'fly away']):
            intent = 'Pro-Migration'
        elif any(k in comm for k in ['nepal mai bas', 'deshmai', 'yahi kei gar', 'farkera', 'return', 'chodne xaina', 'bidesh jana man xaina']):
            intent = 'Anti-Migration'
        else:
            intent = 'Neutral/Observation'
            
    # Fix Primary Driver
    if driver == 'Needs Human Review':
        if is_neta_sarkar:
            driver = 'Systemic/Political Anger'
        elif is_family:
            driver = 'Family Obligation'
        elif is_job_money or 'money' in comm or 'paisa' in comm or 'job' in comm:
            driver = 'Economic Necessity'
        elif 'maya' in comm or 'love' in comm or 'desh ko' in comm:
            driver = 'Patriotism/Love'
        else:
            driver = 'Economic Necessity' # Baseline dominant driver in dataset
            
    # Fix Value Orientation
    if orient == 'Needs Human Review':
        if is_family:
            orient = 'Collectivist-Family'
        elif 'desh' in comm or 'nation' in comm or 'nepali' in comm:
            orient = 'Collectivist-Nation'
        else:
            orient = 'Individualist-Self'
            
    # Fix Affect
    if affect == 'Needs Human Review':
        if any(k in comm for k in ['aasu', 'crying', 'sad', 'regret', 'depressed', 'pida', 'hell', 'struggle', 'lost']):
            affect = 'Despairing/Sad'
        elif any(k in comm for k in ['hope', 'motivate', 'optimistic', 'will change', 'bright', 'rock', 'proud']):
            affect = 'Hopeful/Motivated'
        elif any(k in comm for k in ['wtf', 'fuck', 'shitty', 'chor', 'daka', 'khatey', 'muji', 'nonsense', 'suck', 'angry']):
            affect = 'Angry/Frustrated'
        else:
            affect = 'Pragmatic'
            
    # Post-processing override for meta context rows that don't apply to real migration drivers
    if is_meta or is_lang or is_fact_corr:
        if intent == 'Needs Human Review' or intent == 'Pro-Migration' or intent == 'Anti-Migration':
            if not any(k in comm for k in ['ma janchu', 'desh xadney']):
                intent = 'Neutral/Observation'
        if driver in ['Economic Necessity', 'Systemic/Political Anger', 'Patriotism/Love', 'Family Obligation']:
            if not (is_neta_sarkar or is_job_money or is_family):
                driver = 'Economic Necessity' # Default general classification or keep standard baseline
                
    return pd.Series([intent, driver, orient, affect])

df[['intent', 'primary_driver', 'value_orientation', 'affect']] = df.apply(auto_fix_review, axis=1)

# Check if any instances of 'Needs Human Review' remain
any_left = df.astype(str).apply(lambda x: x.str.contains('Needs Human Review')).any().any()
print(f"Are there any 'Needs Human Review' tags left? {any_left}")

# Save the completely cleared and labeled file
cleared_filename = 'IDS_migration_old_podcast_Cleared.csv'
df.to_csv(cleared_filename, index=False)
print(f"File saved as {cleared_filename}")
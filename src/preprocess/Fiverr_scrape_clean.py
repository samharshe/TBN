import pandas as pd
import os

for file in os.listdir('raw/team/NBA_team'):
    df = pd.read_csv(f'raw/team/NBA_team/{file}')
    df = df.rename(columns={'Match Date': 'DATE', 
                            'Season': 'SEASON',
                            'Game Type': 'COMPETITION',
                            'Pace': 'PACE',
                            'Passes Made': 'PASSES_MADE',
                            'Sec Touch': 'SEC_TOUCH',
                            'Adj Reb Chance %': 'ADJ_REB_CHANCE_%',
                            'Adj Off Reb Chance %': 'ADJ_OFF_REB_CHANCE_%',
                            'Adj Def Reb Chance %': 'ADJ_DEF_REB_CHANCE_%',
                            'Avg Speed Off': 'AVG_SPEED_OFF',
                            'FGM RA': 'FGM_RA',
                            'FGA RA': 'FGA_RA',
                            'FGM Paint': 'FGM_PAINT',
                            'FGA Paint': 'FGA_PAINT',
                            'FGM Mid-Range': 'FGM_MID-RANGE',
                            'FGA Mid-Range': 'FGA_MID-RANGE',
                            'FGM Corner': 'FGM_CORNER',
                            'FGA Corner': 'FGA_CORNER',
                            'FGM AB': 'FGM_AB',
                            'FGA AB': 'FGA_AB',
                            'Deflections': 'DEFLECTIONS',
                            'Screen AST Pts': 'SCREEN_AST_PTS',
                            'Drive Pts': 'DRIVE_PTS',
                            'Catch & Shoot Pts': 'C&S_PTS',
                            'Pull-Up Pts': 'PULL-UP_PTS',
                            'Paint Touch Pts': 'PAINT_TOUCH_PTS',
                            'Post Touch Pts': 'POST_TOUCH_PTS',
                            'Elbow Touch Pts': 'ELBOW_TOUCH_PTS',
                            'Opp 2PT Frequency %': 'OPP_2PT_FREQUENCY_%',
                            'Opp 2PT Field Goals Made': 'OPP_2PT_FGM',
                            'Opp 2PT Field Goals Attempted': 'OPP_2PT_FGA',
                            'Opp 2PT Field Goal %': 'OPP_2PT_FG%',
                            'Opp 3PT Frequency %': 'OPP_3PT_FREQUENCY_%',
                            'Opp 3PT Field Goals Made': 'OPP_3PT_FGM',
                            'Opp 3PT Field Goals Attempted': 'OPP_3PT_FGA',
                            'Opp 3PT Field Goal %': 'OPP_3PT_FG%'
                            })
    
    df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
    df['COMPETITION'] = df['COMPETITION'].map({'Regular Season': 'regular', 'Playoffs': 'playoffs'})
    df = df.round(3)
    
    df.to_csv(f'raw/team/NBA_team2/{file}', index=False)
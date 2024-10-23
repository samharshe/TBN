import time, requests, os
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from io import StringIO
from datetime import timedelta, datetime as dt
from utils import get_team_name_to_TBN_code, get_team_name_to_official_code, get_date_intervals

team_name_to_TBN_code = get_team_name_to_TBN_code()
team_name_to_official_code = get_team_name_to_official_code()
date_intervals = get_date_intervals()

def get_game_links_to_visit(date: dt) -> list[str]:
    """
    visits bbref's "NBA Games Played on {date}" page and returns list of links to the box score pages of games played on that date.
    
    parameters
    -----------
    date  datetime.datetime
        date of the games to be visited.
        
    returns
    -----------
    game_links_to_visit: list[str]
        list of links to box score pages of games played on that date.
    """
    # URL for the "NBA Games Played on {DATE}" page
    date_home_url = f'https://www.basketball-reference.com/boxscores/index.cgi?month={date.month}&day={date.day}&year={date.year}'
    
    # HTML content
    soup = BeautifulSoup(throttle(date_home_url).content, 'html.parser')
    
    # parent element of game links <a> tags
    game_links_td = soup.find_all('td', class_='right gamelink')
    
    # list of game links
    game_links_to_visit = ['https://www.basketball-reference.com' + link.find('a', recursive=False)['href'] for link in game_links_td if link.find('a', recursive=False)]
    
    # return list
    return game_links_to_visit

def minutes_to_decimal(minutes: str | int) -> float:
    """
    takes in (minutes and seconds as string if seconds != 0) else (minutes as int) and standardizes to float rounded to two decimal places.
    parameters
    -----------
    minutes: str
        string,  minutes and seconds
        
    returns
    -----------
    minutes: float
        float rounded to two decimal places.
    """
    # try to convert from string to float
    try:
        minutes, seconds = minutes.split(':')
        return round(float(minutes) + float(seconds) / 60, 2)
    # otherwise, convert from int to float
    except:
        return float(minutes)

def throttle(link: str) -> requests.Response:
    """
    takes in link and returns response from link, chillingout&relaxing for 3 seconds and starting a new session every 20 requests to keep bbref happy.
    
    parameters
    -----------
    link: str
        self-explanatory.
        
    returns
    -----------
    response: requests.Response
        self-explanatory.
    """
    # create first session
    session = requests.Session()
    # track how long current session has been active
    requests_since_last_reset = 0
    
    # reset session every 20 requests
    if requests_since_last_reset >= 20:
        session = requests.Session()
        requests_since_last_reset = 0
    
    # don't make bbref mad
    time.sleep(3)

    # return response...written this way so `throttle(link)` is just smarter version of `session.get(link)`
    return session.get(link)

def get_dfs_from_game_link(game_link: str, date_string: str) -> (pd.DataFrame, pd.DataFrame):
    """
    takes in box score page link and date string and returns two DataFrame objects: one for players and one for teams
    
    parameters
    -----------
    game_link: str
        link to box score page.
    date_string: str
        date of the game in 'YYYY-MM-DD' format. this will be recorded along with all the other data, and this set-up is easier than trying to find the date in the HTML.
        
    returns
    -----------
    players_df: pd.DataFrame
        DataFrame object with all player box-score values, with all empty cells filled with 0 and with date added.
    teams_df: pd.DataFrame
        DataFrame object with all team box-score value, with all values that do not apply to teams removed and with date added.
    """
    # get HTML content of game page
    soup = BeautifulSoup(throttle(game_link).content, 'html.parser')
    
    # find scorebox div
    scorebox = soup.find('div', class_='scorebox')

    # extract div containing team names
    team_divs = scorebox.find_all('div', recursive=False)[:2]  #  first two direct child divs

    # extract team names from the strong tags within these divs
    home = team_divs[1].find('strong').text.strip()
    away = team_divs[0].find('strong').text.strip()

    # convert to the three-letter official codes used in the HTML to TBN and official codes
    home_official_code = team_name_to_official_code[home]
    away_official_code = team_name_to_official_code[away]
    home_TBN_code = team_name_to_TBN_code[home]
    away_TBN_code = team_name_to_TBN_code[away]

    # make DataFrames out of basic and advanced box scores for home and away teams
    x = [('home', home_official_code), ('away', away_official_code)]
    for team, code in x:
        for table_type in ['basic', 'advanced']:
            # make name to assign to DF
            df_name = f'{team}_{table_type}_df'
            # make id to find table
            table_id = f'box-{code}-game-{table_type}'
            # read table into DataFrame
            df = pd.read_html(StringIO(str(soup.find('table', id=table_id))))[0]
            
            # remove multi-level column nonsense
            df.columns = df.columns.get_level_values(-1)
            
            # assign the dataframe to the global namespace
            globals()[df_name] = df
    
    # merge basic and advanced box scores
    home_merged_df = pd.merge(home_basic_df, home_advanced_df, on='Starters', suffixes=('', '_y'))
    away_merged_df = pd.merge(away_basic_df, away_advanced_df, on='Starters', suffixes=('', '_y'))
    
    # remove duplicate columns
    home_merged_df.drop(home_merged_df.filter(regex='_y$').columns, axis=1, inplace=True)
    away_merged_df.drop(away_merged_df.filter(regex='_y$').columns, axis=1, inplace=True)

    # add columns for team and opponent
    home_merged_df['TEAM'] = home_TBN_code
    home_merged_df['OPPONENT'] = away_TBN_code
    away_merged_df['TEAM'] = away_TBN_code
    away_merged_df['OPPONENT'] = home_TBN_code

    # merge dfs
    concat_df = pd.concat([home_merged_df, away_merged_df])

    # rename several columns
    concat_df = concat_df.rename(columns={'Starters': 'PLAYER', '3P': '3PM', 'ORtg': 'ORTG', 'DRtg': 'DRTG', 'GmSc': 'GMSC', 'FTr': 'FTR', '3PAr': '3PAR', 'eFG%': 'EFG%'})
    
    # add column indicating which team is home
    concat_df['HOME_TEAM'] = home_TBN_code

    # add date column
    concat_df['DATE'] = date_string
    
    # for weird games where there is no 'BPM' column
    if 'BPM' not in concat_df.columns:
        concat_df = concat_df.rename(columns={'Unnamed: 16_level_1': 'BPM'})

    # separate 'Team Totals' rows from the main dataframe
    teams_df = concat_df[concat_df['PLAYER'] == 'Team Totals']
    players_df = concat_df[concat_df['PLAYER'] != 'Team Totals']
    
    # remove 'Reserves' row and rows of players not available for the game
    players_df = players_df[players_df['PLAYER'] != 'Reserves']
    players_df = players_df[players_df['MP'] != 'Not With Team']
    players_df = players_df[players_df['MP'] != 'Player Suspended']
    players_df = players_df[players_df['MP'] != 'Did Not Dress']

    # replace 'NaN' and 'Did Not Play' values with 0s
    players_df = players_df.replace({np.nan: 0, 'Did Not Play': 0})
    
    # convert minutes to decimal
    players_df['MP'] = players_df['MP'].apply(minutes_to_decimal)

    # drop columns that make no sense for teams
    teams_df = teams_df.drop(['PLAYER', 'BPM', 'MP', 'USG%', 'GMSC', '+/-'], axis=1)

    # put important columns first
    teams_column_order = ['TEAM', 'DATE', 'OPPONENT', 'HOME_TEAM'] + [col for col in teams_df.columns if col not in ['TEAM', 'OPPONENT', 'HOME_TEAM', 'DATE']]
    teams_df = teams_df[teams_column_order]
    players_column_order = ['PLAYER', 'DATE', 'TEAM', 'OPPONENT', 'HOME_TEAM'] + [col for col in players_df.columns if col not in ['PLAYER', 'TEAM', 'OPPONENT', 'HOME_TEAM', 'DATE']]
    players_df = players_df[players_column_order]

    # return dfs
    return teams_df, players_df

def write_players_df_to_csv(players_df: pd.DataFrame) -> None:
    """
    takes in DataFrame object with player box-score values and writes all values other than 'PLAYER' to 'data/player/bbref_box/{PLAYER}.csv'.
    
    parameters
    -----------
    players_df: pd.DataFrame
        DataFrame object with all team box-score values.
    """
    # group by first column and write CSVs
    for player, data in players_df.groupby(players_df.columns[0]):
        # make filename for player
        filename = os.path.join('data/player/bbref_box', f"{player}.csv")
        
        if os.path.exists(filename):
            # if file exists, append without writing header
            data.iloc[:, 1:].to_csv(filename, mode='a', header=False, index=False)
        else:
            # if file doesn't exist, create it and write header
            data.iloc[:, 1:].to_csv(filename, index=False)

def write_teams_df_to_csv(teams_df: pd.DataFrame) -> None:
    """
    takes in DataFrame object with team box-score values and writes all values other than 'TEAM' to 'data/team/bbref_box/{TEAM}.csv'.
    
    parameters
    -----------
    teams_df: pd.DataFrame
        DataFrame object with all team box-score values.
    """
    # group by first column and write CSVs
    for team, data in teams_df.groupby(teams_df.columns[0]):
        # make filename for player
        filename = os.path.join('data/team/bbref_box', f"{team}.csv")
        
        # if file exists, append without writing the header
        if os.path.exists(filename):
            data.iloc[:, 1:].to_csv(filename, mode='a', header=False, index=False)
        # if file doesn't exist, create it and write the header
        else:
            data.iloc[:, 1:].to_csv(filename, index=False)

def main():
    for start_date, end_date in date_intervals:
        # set start date
        y, m, d = start_date
        current_date = dt(y, m, d)
        
        # name changes at beginning of 2014 season...done this way to distinguish 2014- Hornets from -2001 Hornets
        global team_name_to_TBN_code
        global team_name_to_official_code
        
        if y == 2014:
            team_name_to_TBN_code['Charlotte Hornets'] = 'HOR'
            team_name_to_official_code['Charlotte Hornets'] = 'CHO'
        
        # set end date
        y, m, d = end_date
        end_date = dt(y, m, d)
        
        # iterate through all dates between start and end dates
        while current_date <= end_date:
            # date string
            date_string = current_date.strftime('%Y-%m-%d')
            # keep the console chatty
            print('GETTING DATA FROM GAMES PLAYED ON:', date_string)
            # URL for the "NBA Games Played on {DATE}" page
            game_links_to_visit = get_game_links_to_visit(current_date)

            # iterate through list of links
            for game_link_to_visit in game_links_to_visit:
                # keep in touch with the people
                print('GETTING DATA FROM LINK:', game_link_to_visit)
                # get dfs
                teams_df, players_df = get_dfs_from_game_link(game_link_to_visit, date_string)
                # write dfs
                write_players_df_to_csv(players_df)
                write_teams_df_to_csv(teams_df)
                
            # clear console
            os.system('cls' if os.name == 'nt' else 'clear')
                
            # today -> tomorrow
            current_date += timedelta(days=1)
    
if __name__ == '__main__':
    main()
# TODO docstrings
import os, asyncio, random
from playwright.async_api import async_playwright
from datetime import datetime, date, timedelta
import pandas as pd
from utils import get_team_name_to_TBN_code, get_stretches

from bs4 import BeautifulSoup
from IPython.display import display, clear_output

async def get_df(url: str, stat_type: str) -> pd.DataFrame:
    """
    create and return DataFrame corresponding to URL and stat type passed as parameters.
    
    
    stat_type is passed as parameter to make logic easier, since different stat types have pages that need to be parsed differently.
    it would be annoying to rummage through URL to determine which stat it's for.
    
    parameters
    ----------
    url: str
        self-explanatory.
    stat_type: str
        self-explanatory.
        
    returns
    ----------
    df: pd.DataFrame
        DataFrame corresponding to URL and stat type passed as parameters.
        which statistics to glean from each stat type is hard-coded internally in this function.
    """
    # get table from HTML
    table = await get_table(url)
    
    # get and clean headers
    headers = [th.text.strip().upper().replace(u'\xa0', u' ') for th in table.find_all('th') if not th.text.strip().endswith('RANK')]
    
    # deal with multi-level headers for shooting by zone
    if stat_type == 'shooting?DistanceRange=By+Zone':
        headers = headers[10:]
        prefixes = ['RA_', 'PAINT_NON_RA_', 'MID-RANGE_', 'LEFT_CORNER_3', 'RIGHT_CORNER_3', 'CORNER_3', 'ABOVE_BREAK_3']
        new_headers = []
        for prefix in prefixes:
            new_headers.extend([f"{prefix}{header}" for header in headers[:3]])
        headers = ['TEAM'] + new_headers
        
    # make list of table rows
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        row = [td.text.strip() for td in tr.find_all('td')]
        rows.append(row)
        
    # make table into df
    df = pd.DataFrame(rows, columns=headers)
    
    # select only certain columns for each stat type
    if stat_type == 'advanced':
        df = df[['TEAM', 'PIE', 'PACE']]
    elif stat_type == 'passing':
        df = df[['TEAM', 'PASSESMADE']]
        df.columns = ['TEAM', 'PASSES_MADE']
    elif stat_type == 'touches':
        df = df[['TEAM', 'AVG SEC PERTOUCH']]
        df.columns = ['TEAM', 'SEC_TOUCH']
    elif stat_type == 'rebounding':
        df = df[['TEAM', 'ADJUSTEDREB CHANCE%']]
        df.columns = ['TEAM', 'ADJ_REB_CHANCE_%']
    elif stat_type == 'offensive-rebounding':
        df = df[['TEAM', 'ADJUSTEDOREB CHANCE%']]
        df.columns = ['TEAM', 'ADJ_OFF_REB_CHANCE_%']
    elif stat_type == 'defensive-rebounding':
        df = df[['TEAM', 'ADJUSTEDDREB CHANCE%']]
        df.columns = ['TEAM', 'ADJ_DEF_REB_CHANCE_%']
    elif stat_type == 'speed-distance':
        df = df[['TEAM', 'AVG SPEED OFF']]
        df.columns = ['TEAM', 'AVG_SPEED_OFF']
    elif stat_type == 'shooting?DistanceRange=By+Zone':
        df = df[[col for col in df.columns if not ('LEFT' in col or 'RIGHT' in col or '%' in col)]]
    elif stat_type == 'hustle':
        df = df[['TEAM', 'DEFLECTIONS', 'SCREENASSISTS PTS']]
        df.columns = ['TEAM', 'DEFLECTIONS', 'SCREEN_ASSIST_PTS']
    elif stat_type == 'shooting-efficiency':
        df = df[['TEAM', 'DRIVEPTS', 'C&SPTS', 'PULL UPPTS', 'PAINTTOUCH PTS', 'POSTTOUCH PTS', 'ELBOWTOUCH PTS']]
        df.columns = ['TEAM', 'DRIVE_PTS', 'C&S_PTS', 'PULL_UP_PTS', 'PAINT_TOUCH_PTS', 'POST_TOUCH_PTS', 'ELBOW_TOUCH_PTS']
        
    # return df
    return df

def get_url(stat_type: str, season_type: str, season: str, formatted_date: str) -> str:
    """
    create and return NBA.com URL based on query string inputs.
    
    parameters
    ----------
    stat_type: str
        self-explanatory.
    season_type: str
        "Regular+Season", "Playoffs", or "PlayIn"
    season: str
        YYYY-YY format, as in 2023-24.and
    formatted_date: str
        date in format MM%2FDD%2FYYYY, as in 01%2F01%2F2024.
        
    returns
    ----------
    url: str
        NBA.com URL based on query string inputs.
    """
    # makes it easier to deal with fact that some stat types have query strings and others do not, without clogging up other functions with logic
    if '?' in stat_type:
        url = f'https://www.nba.com/stats/teams/{stat_type}&SeasonType={season_type}&Season={season}&DateFrom={formatted_date}&DateTo={formatted_date}'
    else:
        url = f'https://www.nba.com/stats/teams/{stat_type}?SeasonType={season_type}&Season={season}&DateFrom={formatted_date}&DateTo={formatted_date}'
        
    return url

async def dynamic_delay() -> None:
    """
    waits for random amount of time between 2 and 5 and additional random amount of time between 0 and 2 with probability 0.2.
    """
    base_delay = random.uniform(2, 5)
    extra_delay = random.uniform(0, 2) if random.random() < 0.2 else 0
    await asyncio.sleep(base_delay + extra_delay)

async def get_html(url: str) -> str:
    """
    """
    # BrightData credentials
    SBR_WS_CDP = 'xxx'
    
    async with async_playwright() as pw:
        # connect to BrightData's headless browser
        browser = await pw.chromium.connect_over_cdp(SBR_WS_CDP)
        # return html content at url via BrightData browser
        try:
            page = await browser.new_page()
            await page.goto(url) # often throws TimeoutError after NBA.com stops fing w us
            html = await page.content()
            return html
        finally:
            # it might be slow to create and destroy browser each time...not sure
            await browser.close()

async def get_table(url: str) -> BeautifulSoup:
    """
    """
    # get html  
    html = await get_html(url)
    
    # parse html
    soup = BeautifulSoup(html, 'html.parser')
    
    # find table
    table = soup.find('table', class_='Crom_table__p1iZz') # this returns None if no table is found, which is often the case when NBA.com realizes these are bot requests
    
    # throw error here for simpler debugging
    if table is None:
        raise Exception(f'No table found for URL: {url}')
    
    # if no error, return table
    return table
        
def process_df_list(df_list: list, current_date: date, season: str, season_type: str) -> pd.DataFrame:
    """
    """
    # turn df_list into df
    df = pd.concat(df_list, axis=1)
    # remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # add date column
    df['DATE'] = current_date.strftime('%Y-%m-%d')
    df['SEASON'] = season
    df['GAME_TYPE'] = season_type
    
    # move date, season, and game type to the front
    cols = ['DATE', 'SEASON', 'GAME_TYPE']
    df = df[cols + [col for col in df.columns if col not in cols]]
    
    # return df
    return df

def write_df(df: pd.DataFrame):
    """
    writes team stats DataFrame to CSV file, creating new file if it doesn't exist.
    
    parameters
    ----------
    df: pd.DataFrame
        self-explanatory.
    """
    # iterate through each row in df
    for _, row in df.iterrows():
        # get team name code
        team = team_name_to_TBN_code[row['TEAM']]
        # create filename
        filename = f'../data/team/NBA/{team}.csv'
        # get data values as df
        row_df = pd.DataFrame([row.values], columns=df.columns)
        # drop team column since this information is saved in filename
        row_df = row_df.drop(columns=['TEAM'])
        # if file already exists, append row to it
        if os.path.exists(filename):
            row_df.to_csv(filename, mode='a', header=False, index=False)
        # if not, make new file, write headers, and write row
        else:
            row_df.to_csv(filename, index=False)

async def process_stretch(stretch: tuple, stat_types: list) -> None:
    """
    """
    # unpack tuple
    start_date, end_date, season, season_type = stretch
    
    # start with first date in stretch
    current_date = start_date
    
    # iterate through each date in the stretch
    while current_date <= end_date:
        # formatted date for url
        formatted_date = current_date.strftime('%m%%2F%d%%2F%Y')
        
        # get df for each stat type, to be concatenated into one df later
        df_list = []
        
        # iterate through stat_types list
        for stat_type in stat_types:
            # get url based on query string inputs
            url = get_url(stat_type, season_type, season, formatted_date)

            # use get_df to clean and format table
            df = await get_df(url, stat_type)
            
            # display df
            print(f"{stat_type} DataFrame:")
            display(df)
            
            # append df to df_list
            df_list.append(df)
            
        # turn df_list into single, tidy df
        df = process_df_list(df_list, current_date, season, season_type)
        
        # display df
        print(f'{current_date.strftime("%Y-%m-%d")} DataFrame:')
        display(df)
        
        # write df to csv
        write_df(df)

        # ungunkify console
        clear_output()
                
        # today -> tomorrow
        current_date += timedelta(days=1)

async def main():
    # get stretches from utils 
    stretches = get_stretches()
    
    # get team name codes from utils
    team_name_to_TBN_code = get_team_name_to_TBN_code()
    # manually set team name codes for teams that have changed names
    team_name_to_TBN_code['Charlotte Hornets'] = 'HOR'
    team_name_to_TBN_code['LA Clippers'] = 'CLI'

    # list types of data to scrape
    stat_types = [
        'advanced',
        'passing',
        'touches',
        'rebounding',
        'offensive-rebounding',
        'defensive-rebounding',
        'speed-distance',
        'shooting?DistanceRange=By+Zone',
        'hustle',
        'shooting-efficiency',
    ]
    
    # iterate through stretches...this line easily modified to scrape data for multiple stretches concurrently
    for stretch in stretches:
        await process_stretch(stretch, stat_types)
        
if __name__ == '__main__':
    asyncio.run(main())
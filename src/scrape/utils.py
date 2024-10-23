from datetime import date

def get_team_name_to_TBN_code() -> dict:
    """
    get map from team name to code used throughout this repo. this scheme preferred over official scheme for its consistency.
    
    returns:
        dict: map from team name to code. by default, all codes are first 3 characters of mascot. 
            example: 'Detroit Pistons' -> 'PIS'.
        for discontinued teams whose mascot was later taken up by another team, the code is modified using the city name
            example: 'Vancouver Grizzlies' -> 'VGR'.
        NB that this lacks modern code for Charlotte Hornets. when season >= 2014-15, this should be overwritten with 'Charlotte Hornets' -> 'HOR' to distinguish modern from old Hornets.
            in this dict: 'Charlotte Hornets' -> 'CHO'.
            in seasons >= 2014-15 (requires overwrite): 'Charlotte Hornets' -> 'HOR'.
    """
    return {
        'Atlanta Hawks': 'HAW',
        'Boston Celtics': 'CEL',
        'Brooklyn Nets': 'NET',
        'Chicago Bulls': 'BUL',
        'Cleveland Cavaliers': 'CAV',
        'Dallas Mavericks': 'MAV',
        'Denver Nuggets': 'NUG',
        'Detroit Pistons': 'PIS',
        'Golden State Warriors': 'WAR',
        'Houston Rockets': 'ROC',
        'Indiana Pacers': 'PAC',
        'Los Angeles Clippers': 'CLI',
        'LA Clippers': 'CLI',
        'Los Angeles Lakers': 'LAK',
        'Memphis Grizzlies': 'GRI',
        'Miami Heat': 'HEA',
        'Milwaukee Bucks': 'BUC',
        'Minnesota Timberwolves': 'TIM',
        'New Orleans Pelicans': 'PEL',
        'New York Knicks': 'KNI',
        'Oklahoma City Thunder': 'THU',
        'Orlando Magic': 'MAG',
        'Philadelphia 76ers': '76E',
        'Phoenix Suns': 'SUN',
        'Portland Trail Blazers': 'BLA',
        'Sacramento Kings': 'KIN',
        'San Antonio Spurs': 'SPU',
        'Toronto Raptors': 'RAP',
        'Utah Jazz': 'JAZ',
        'Washington Wizards': 'WIZ',
        'New Orleans Hornets': 'NHO',
        'New Orleans/Oklahoma City Hornets': 'NOH',
        'Seattle SuperSonics': 'SUP',
        'New Jersey Nets': 'NNE',
        'Charlotte Bobcats': 'BOB',
        'Charlotte Hornets': 'CHO',
        'Vancouver Grizzlies': 'VGR'
    }

def get_team_name_to_official_code() -> dict:
    """get official map from  team name to code, used on NBA.com, bbref.com, etc.
    
    returns:
        dict: map from team name to official code. unfortunately, there is no consistency in this scheme.
            examples: 'Detroit Pistons' -> 'DET'; 'Brooklyn Nets' -> 'BRK'; 'New York Knicks' -> 'NYK'.
    """
    return {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS',
        'Brooklyn Nets': 'BRK',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'Los Angeles Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHO',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS',
        'New Orleans Hornets': 'NOH',
        'New Orleans/Oklahoma City Hornets': 'NOK',
        'Seattle SuperSonics': 'SEA',
        'New Jersey Nets': 'NJN',
        'Charlotte Bobcats': 'CHA',
        'Charlotte Hornets': 'CHH',
        'Vancouver Grizzlies': 'VAN'
    }

def get_date_intervals() -> list:
    """get list of tuples, where each tuple is start date and end date for a stretch of games. this is usually, but not always, a season. this is to avoid iterating through dead periods.
    
    returns:
        list: list of tuples, where each tuple is start date and end date for a stretch of games.
            examples: `((2000, 10, 31), (2001, 6, 15))` is 2000-01 season; `((2020, 7, 31), (2020, 10, 11)))` is games played in 2019-20 bubble.
    """
    return [
        ((2000, 10, 31), (2001, 6, 15)),
        ((2001, 10, 30), (2002, 6, 12)),
        ((2002, 10, 29), (2003, 6, 15)),
        ((2003, 10, 28), (2004, 6, 15)),
        ((2004, 11, 2), (2005, 6, 23)),
        ((2005, 11, 1), (2006, 6, 20)),
        ((2006, 10, 31), (2007, 6, 14)),
        ((2007, 10, 30), (2008, 6, 17)),
        ((2008, 10, 28), (2009, 6, 14)),
        ((2009, 10, 27), (2010, 6, 17)),
        ((2010, 10, 26), (2011, 6, 12)),
        ((2011, 12, 25), (2012, 6, 21)),
        ((2012, 10, 30), (2013, 6, 20)),
        ((2013, 10, 29), (2014, 6, 15)),
        ((2014, 10, 28), (2015, 6, 16)),
        ((2015, 10, 27), (2016, 6, 19)),
        ((2016, 10, 25), (2017, 6, 12)),
        ((2017, 10, 17), (2018, 6, 8)),
        ((2018, 10, 16), (2019, 6, 13)),
        ((2019, 10, 22), (2020, 3, 11)),
        ((2020, 7, 31), (2020, 10, 11)),
        ((2020, 12, 22), (2021, 7, 20)),
        ((2021, 10, 19), (2022, 6, 16)),
        ((2022, 10, 18), (2023, 6, 12)),
        ((2023, 10, 24), (2024, 6, 17))
    ]
    
def get_stretches() -> list:
    """get list of tuples, where each tuple is start date and end date for a stretch of games, along with season and season type (regular season, playoffs, play-in) to which that stretch belongs. structured for use by NBA.com scraper (NBA.py).
    
    returns:
        list: list of tuples, where each tuple is start date, end date, season, season type.
            examples: `(date(2013, 10, 29), date(2014, 4, 16), '2013-14', 'Regular+Season')` is 2013-14 regular season; `(date(2020, 8, 15), date(2020, 8, 16), '2019-20', 'PlayIn')` is 2019-20 play-in games.
    """
    return [
        (date(2015, 10, 27), date(2016, 4, 13), '2015-16', 'Regular+Season'),
        (date(2016, 4, 16), date(2016, 6, 19), '2015-16', 'Playoffs'),
        
        (date(2016, 10, 25), date(2017, 4, 12), '2016-17', 'Regular+Season'),
        (date(2017, 4, 15), date(2017, 6, 12), '2016-17', 'Playoffs'),
        
        (date(2017, 10, 17), date(2018, 4, 11), '2017-18', 'Regular+Season'),
        (date(2018, 4, 14), date(2018, 6, 8), '2017-18', 'Playoffs'),
        
        (date(2018, 10, 16), date(2019, 4, 10), '2018-19', 'Regular+Season'),
        (date(2019, 4, 13), date(2019, 6, 13), '2018-19', 'Playoffs'),
        
        (date(2019, 10, 22), date(2020, 3, 11), '2019-20', 'Regular+Season'),
        (date(2020, 7, 31), date(2020, 8, 14), '2019-20', 'Regular+Season'),
        (date(2020, 8, 15), date(2020, 8, 16), '2019-20', 'PlayIn'),
        (date(2020, 8, 17), date(2020, 10, 11), '2019-20', 'Playoffs'),
        
        (date(2020, 12, 22), date(2021, 5, 16), '2020-21', 'Regular+Season'),
        (date(2021, 5, 18), date(2021, 5, 21), '2020-21', 'PlayIn'),
        (date(2021, 5, 22), date(2021, 7, 20), '2020-21', 'Playoffs'),
        
        (date(2021, 10, 19), date(2022, 4, 10), '2021-22', 'Regular+Season'),
        (date(2022, 4, 12), date(2022, 4, 15), '2021-22', 'PlayIn'),
        (date(2022, 4, 16), date(2022, 6, 16), '2021-22', 'Playoffs'),
        
        (date(2022, 10, 18), date(2023, 4, 9), '2022-23', 'Regular+Season'),
        (date(2023, 4, 11), date(2023, 4, 14), '2022-23', 'PlayIn'),
        (date(2023, 4, 15), date(2023, 6, 12), '2022-23', 'Playoffs'),
        
        (date(2023, 10, 24), date(2024, 4, 14), '2023-24', 'Regular+Season'),
        (date(2024, 4, 16), date(2024, 4, 19), '2023-24', 'PlayIn'),
        (date(2024, 4, 20), date(2024, 6, 17), '2023-24', 'Playoffs')
    ]

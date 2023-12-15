from sentify.config.configuration import TweetScraperConfig

from ntscraper import Nitter 


class TweetScraper:
    def __init__(self, config: TweetScraperConfig):
        '''
        creates an instance for scraping tweets using Ntscraper
        '''
        self.config = config
        self.scraper = Nitter(log_level=self.config.log_level, 
                              skip_instance_check=self.config.skip_instance_check)
        
    def scrape_tweets(self, query: str, mode: str, number: int):
        '''
        scrapes tweets for provided query in the given mode
        
        ## Parameters:
        
        query: str
            the query which needs to be search on the twitter
            
        mode: str
            mode of search, defines where query belongs to like user or hashtag
            
        number: int
            the number of tweets which needs to be scraped 
        '''
        scraped_tweets = self.scraper.get_tweets(query, mode, number)
        
        return scraped_tweets['tweets']
    
from sentify.components.tweet_scraper import TweetScraper
from sentify.config.configuration import ConfigurationManager
from logger import logger 


class Scraper:
    def __init__(self):
        '''
        creates the scraper class for scraping related content
        '''
        self.config = ConfigurationManager()
        self.tweet_scraper_config = self.config.get_tweet_scraper_config()
        self.__initiate_server()
        
    def scrape_tweets(self, query: str, mode: str, number: int = 10):
        '''
        scrapes tweets using Ntscraper from Nitter
        
        ## Parameters:
        
        query: str
            the query which needs to be searched
        
        mode: str
            mode of search, defines the nature of query like user or hashtag
        
        number: int or 10
            the maximum number of tweets to be scraped
        '''
        tweets = self.scraper.scrape_tweets(query, mode, number)
        logger.info("Tweets Scraped!!!")
        
        return tweets
        
    def __initiate_server(self):
        try:
            logger.info("Starting Ntscraper server")
            self.scraper = TweetScraper(self.tweet_scraper_config)
            logger.info("Server initiated")
        except Exception as e:
            logger.exception(e)
            raise e 
        
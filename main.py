import scrapy

class TopViewsSpider(scrapy.Spider):
    name = "topviews"
    start_urls = [
        "https://pageviews.wmcloud.org/topviews/?project=en.wikipedia.org&platform=all-access&date=2022&excludes="
    ]

    def parse(self, response):
        for tr in response.css("tbody.topview-entries tr.topview-entry"):
            yield {
                "rank": tr.css("td.rank::text").get(),
                "title": tr.css("td.title a::text").get(),
                "views": tr.css("td.views::text").get(),
                "link": tr.css("td.topview-entry--label a::href").get()
            }

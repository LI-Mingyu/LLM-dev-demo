import scrapy

class QaSpider(scrapy.Spider):
    name = "qa"
    allowed_domains = ["help.aliyun.com"]
    start_urls = ["https://help.aliyun.com/zh/ecs/support/faq-3"]
    custom_settings = {
        'DEPTH_LIMIT': 3,
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.RFPDupeFilter'
    }

    def parse(self, response):
        # 检查响应的深度
        depth = response.meta.get('depth', 0)
        if depth < 3:
            # 提取页面中的链接
            for href in response.css('a::attr(href)').extract():
                yield response.follow(href, self.parse)
        
        # 提取页面的内容
        yield {
            'url': response.url,
            'content': response.css('body').get(),
        }

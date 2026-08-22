import scrapy

class QaSpider(scrapy.Spider):
    name = "qa"
    allowed_domains = ["help.aliyun.com"]
    start_urls = ["https://help.aliyun.com/zh/ecs/support/troubleshooting-1/"]

    def parse(self, response):
        # 提取<section class="aliyun-docs-view" id="aliyun-docs-view">标签内的内容
        section_content = response.xpath('//*[@id="aliyun-docs-view"]').get()
        if section_content:
            # 提取该标签内的链接并跟踪
            for href in response.xpath('//*[@id="aliyun-docs-view"]//a/@href').extract():
                if href.startswith('/zh/ecs/support/'):
                    yield response.follow(href, self.parse)
            
            # 提取页面的内容
            yield {
                'url': response.url,
                'content': section_content,
            }

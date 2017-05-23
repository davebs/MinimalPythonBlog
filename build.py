import jinja2
import markdown
import os

SITE_PREFIX = ''

# list posts
post_filenames = sorted(os.listdir('posts'))

post_listing_contexts = []

# load post template, we'll render 1 for each post found in posts directory
post_template_filename = 'post.html'
loader = jinja2.FileSystemLoader('templates/')
template = jinja2.Environment(loader=loader).get_template(post_template_filename)

for post_filename in post_filenames:
    lines = open('posts/'+post_filename).readlines()
    content_html = markdown.markdown(''.join(lines), 
            extensions=['markdown.extensions.codehilite', 'markdown.extensions.fenced_code'])

    # we'll use these to render the index page after writing out each post
    post_listing_context = {
                'title': lines[0][1:],
                'url': SITE_PREFIX+post_filename[:-2]+'.html'
            }
    post_listing_contexts.append(post_listing_context)

    post_content_context = {
                'content': content_html
            }
    rendered_post = template.render(post_content_context)
    with open("build/"+post_filename[:-2]+'.html', "wb") as fh:
            fh.write(rendered_post)

# render index
filename = 'index.html'
loader = jinja2.FileSystemLoader('templates/')
index = jinja2.Environment(loader=loader).get_template(filename)

# build the index page
rendered_index = index.render({'posts': post_listing_contexts})
with open('build/index.html', "wb") as fh:
        fh.write(rendered_index)

# upload to s3
# aws_bucket = boto.aws_bucket('lsdkjfldskfj')
# upload_dir('build', aws_bucket)

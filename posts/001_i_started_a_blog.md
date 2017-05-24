# I wanted to start blogging so I made a static site generator

After years of arguing with people over the internet on twitter and assorted message boards, I've finally decided my thoughts are important enough to immortalize in the form of a "web log" (colloquially known as a blog.)

Since it's 2017, there are at least 100 ways to easily setup a blog and start blogging. But why make it easy? 

### With a static site generator, you write stuff down, then run a script to sorta compile things into a series of HTML files which can then be uploaded anywhere (like amazon S3). The upside of this is that pages load blindingly fast because there's no hit on any server backend when you request a page. For instance, when you visit a wordpress page, there's a whole bunch of things that need to happen before you get your page. Not so with static websites.

I mean, sure, I just want to write stuff down and put it on the internet. I could sign up for Medium and instantly tap into a network of other self-obsessed tech types. I could get a wordpress site and install tons of plugins to boost my SEO and other such shenanigans. I could paste stream-of-consciousness nonsense into a pastebin and distribute the link.

But since I'm kind of a hipster software developer, a "blogging platform" simply isn't going to cut it. No, I want something more minimal. Something untested, with almost no features. Something like a static site generator.

I've heard good things about Jekyll. But why use something that's perfectly good and widely supported when **I can make my own**!

So I did. I used it to publish this site. I put the code on github <a href="https://github.com/davebs/MinimalPythonBlog">here</a>

It's pretty much the most minimal static site generator for blogging I can conceive of. It's all in python. It uses jinja2 to render templates. It scans a directory for text files written in markdown, then converts that markdown into HTML (w/ python markdown), adds syntax coloring for code (with pygments), and updates the index page listing posts.

**Moral of the story? Python is awesome.**

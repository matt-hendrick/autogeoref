_Everything in this doc is human written._

## Why I Made This

This project aims to help speed up the process of georeferencing old Sanborn fire insurance maps. Those maps are utilized by researchers and provide useful information on how cities and places have changed over time. While some of the maps have been digitized as PDFs, only a subset of those have been georeferenced to a specific location. That lack of georeferencing makes the process of using Sanborn maps cumbersome and slow. It also impedes researchers who use georeferenced Sanborn maps as inputs.

I got interested in this topic for three reasons:

1. I volunteer for a local advocacy group called Strong Towns Chicago. One of our members used Sanborn maps as a tool to help visually communicate what could be done with our streets and to illustrate the impacts of previous policies on our neighborhoods ([article](https://www.strongtownschicago.org/articles/3-ways-to-make-this-chicago-megaproject-success-everyone)).
2. I saw a University of Chicago project ([Chicago Urban Heritage Project](https://chicagourbanheritage.com/)) that used Sanborn maps to analyze neighborhoods on Chicago's south side. Their work was used by [a proposed housing project to illustrate that their proposed density was something the neighborhood had already had in the 1920s](https://www.cbsnews.com/chicago/news/uchicago-students-chicago-urban-heritage-project/). They fed manually georeferenced maps to a computer-vision pipeline that extracted building footprints.
3. Adam Cox's website [OldInsuranceMaps.net](https://oldinsurancemaps.net) and [his amazing open source tool](https://github.com/ohmg-dev/OldInsuranceMaps) allow people to manually georeference Sanborn maps.

## Current State of Georeferencing Sanborn Maps

For both the U Chicago and Adam Cox processes I noticed there was a lot of time-consuming manual work involved in georeferencing each page of a Sanborn map. That manual work is a bottleneck to research projects focused on cities that rely on georeferenced maps as inputs. As a February 2025 paper states, ["using traditional manual georeferencing workflows on large map collections can be prohibitively time consuming and expensive"](https://www.tandfonline.com/doi/full/10.1080/15420353.2025.2462737). Because of that, large portions of the extant digitized Sanborn map collection are not georeferenced. Adam Cox's Old Insurance Maps site has done great work in making that process easier, but that still requires significant manual labor.

## What This Project Does

A multi-modal model is provided with an image of a Sanborn map => spits out the street names, railroads, and house numbers it sees in JSON => that JSON is fed to deterministic pipeline => deterministic pipeline attempts to match it to the correct streets or rail lines in the current city.

As far as I am aware, the best performing automated approach to automating Sanborn map georeferencing prior to this approach was able to automatically place 14% of sheets ([using an object detection model](https://www.tandfonline.com/doi/full/10.1080/15420353.2025.2462737)). This approach can place more than 70% of sheets and those placements generally score well when compared to manual human placements (median difference of 5.4 meter from human placement with only 11% greater than 15 meters different). The output of this system generally just looks good on a map (and it has improved as I have iterated on it).

### Scored Accuracy Rates vs OldInsuranceMaps pins

- Placed ~78% of 3,000+ sheets on the 35 Chicago volumes that had OldInsuranceMaps pins
- Placed ~75% of sheets across the digitized Chicago corpus
- For volumes with placeable sheets, sheet placement rate varied a lot by volume (from 2% to 97%).
- Running the pipeline against a Cleveland volume, the system placed 64% of sheets

My hope is that this could be a useful tool that helps to speed up the process of georeferencing these maps and to facilitate the research that depends on them. It can provide humans with a solid basis of reasonably well placed sheets, instead of researchers having to start from scratch. The project has code that allows for users to manually place pins or edit existing sheet placement, but I think OldInsuranceMaps already does an excellent job at this.

### Limitations

The current system is not perfect. It attempts to place sheets to their real world locations, fit the sheet to the current street grid, and intelligently handle sections in which two sheets overlap. However, some sheets are slightly misaligned and some boundaries between sheets are jagged. Sometimes sheets can be placed incorrectly. I expect additional improvements to the algorithm that determines how sheets are validated, fitted, and warped could reduce the frequency and severity of those issues. I was starting to see diminishing returns on my repeated tweaking of that stage of the process and I wanted to release something useable without holding out for perfection. In some cases, significant differences in the modern street grid or quirks of the old, hand-drawn map sheets make it unlikely that a generic deterministic algorithm could precisely place all sheets without error. There are some parts of cities that will inherently be more difficult to place (areas like parks, rivers, or lakes where there are not many street labels to correlate).

This (unsurprisingly) struggles if parts of the city grid have changed significantly since the map was created (highways destroying neighborhoods, water being filled in, streets being changed or removed). It also struggles on sheets that are mostly parkspace or water (where there are no streets or rail lines to place). It works best on clean grids and on locations that have not seen major modifications to their street grid.

I had hoped to be able to use local open source models to annotate the sheets, but the open source models I tried (Qwen 3.5, Gemma 4, Minicpm-v4.5) did not perform well on this task. Ideally in the future, smaller models can do this on people's machines without having to hit an external API. For the model calls, the system supports either hitting the OpenAI/Anthropic APIs OR shelling out to Claude Code/Codex/OpenCode.

---

## Notes on the Experience of Building this With Coding Agents

LLMs and coding agents were very helpful in 1) figuring out a mechanism to automate georeferencing of some Sanborn sheets, 2) iteratively improving on that process, and 3) allowing me to quickly build something in a domain that I did not have expertise in (although I am certain someone with more GIS expertise would be able to move more quickly with them).

Fable and Opus 5 are pretty good at taking a bunch of theories and hypotheses and stress testing them. This pipeline has gradually improved its outputs through many rounds of me coming up with ideas and then having models attempt to prove them wrong. In many cases, the models suggested variants of those ideas that have ended up bearing fruit. The vast majority of ideas did not prove out, but seeing parallel agents test out alternate hypotheses on worktrees and then being able to view that there were more sheets placed on the map was interesting.

Having models generate HTML to explain a concept or idea was quite helpful. This pipeline has evolved to include many stages and at a certain point, I was losing track of the various stages. Having a model generate a [visual walkthrough of the algorithm](https://autogeoref.com/viewer/walkthrough.html) was helpful. It took many iterations to generate that walkthrough as it repeatedly generated incorrect representations.

Prior to coding agents, I would never have created the insane level of static analysis/linting that this project has. But each of those rules was in response to agents repeatedly doing something I disliked. Eventually, I got to a point where if an agent did something I did not like, I had it adopt an off-the-shelf rule or write a custom static analysis check to programmatically prevent that. The project's current static analysis checks do the following:

- Hard caps on docstring and comment length
- From comments and docstrings, banned dates, specific Sanborn map identifiers, specific words that indicate the agent was writing some comment specific to a single city (Chicago, Cleveland, the Loop).
- Set a hard cap of lines per file, public names per file.
- A check for dead code that isn't reachable from another part of the core code (tests don't count). All code must be reachable from the core CLI.
- Hard caps on the number of imports per file.
- Rules around which layers can import other layers (to help keep Claude from breaking down seams in the codebase). Pipeline stages are banned from importing one another, for example.
- Checks to validate heavy imports are not imported into lightweight sections of the codebase.
- Tests are banned from making http calls or model calls.

I ran into the following issues with using agents for this (most of which are not surprising to anyone who has used coding agents):

- Claude initially decided to associate all of my commits with my personal gmail address. It also chose to intersperse that gmail address throughout the codebase ("for attribution"). Scrubbing that was the primary reason I have squashed the many, many commits this project had prior to release.
- The model would respond to a mistake or issue by silently encoding a principle in a comment or documentation. Those principles did not always align with my actual intent. Later on, future agents would make decisions based upon those warped comments and I had to correct them/rip out the false guidance.
- Fable was consistently better than Opus 4.8. Opus 4.8 blew up my CLAUDE.md and documentation with nonsense. However, when Opus 5.0 came out, I switched to it and have had good results.
- Files and methods were growing unbounded. Without me telling an agent to decompose a file or method, it tended to grow.
- At one point, an agent decided it needed to take many, many high resolution screenshots to complete its task. That filled up my disk, killed all in-process agents, and temporarily locked up WSL on my machine.
- Many, many times Claude has confidently told me something was implemented perfectly and I noticed a clear bug. In many earlier iterations, the automated pipeline placed sheets on the map that were clearly in the wrong spot or had some terrible visual artifact. Even when I gave Claude tools to visually inspect the sheet, it would assert everything was great where there were clear and obvious visual flaws.

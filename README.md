branch manager
rop page


** templates file -> index.html **
also pictures or other requiered assets are placed in static file 
1) an upload page ( the user can upload a picture here)
2) after submiting the user will be redirected to the results page ( + it may take a while so you should put a spinner, loader bar, or anything else that you think is appropriate)
3) next, there is a page that carry results ( now both the upload page and results page place in index.html )
4) results page has several sections
4.1) the first part is the handeling of segmented eye picture( it comes from jinja)
4.2) then, beneath the segmented eye pictuerr is 4 parts or boxes for text-based results that the pipeline had processed(plus, stage, and zone prediction, as well as final decision)( each of them has a percentage as their accuracy + final decision has a text for treatment) 
4.3) a section for anotation by real docotors. ( this part has three drop down + a box for more explanation)
4.4) final section is a chatbot that the user can talk with our model

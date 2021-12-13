function generate_ngrams_list(group, ngrams_method){
    //fetch('http://127.0.0.1:5000/api/characterization/'+group+'/'+ngrams_method)
    fetch('http://134.59.134.227:8080/api/'+group+'/'+ngrams_method)
    .then(async function(response) {
        const text = await response.text();
        console.log(text);
        data = JSON.parse(text);
        var json_file_words = data['json']

        document.getElementById('json_file_words').value = ''
        document.getElementById('json_file_words').value = json_file_words
        
        document.getElementById("community_title").innerHTML = data['head'];

        var ul = document.getElementById("ngrams");
        ul.innerHTML = "";

        for (let i of data['ngrams']) {
            let li = document.createElement("li");
            li.innerHTML = i;
            ul.appendChild(li);
        }

    })
    
}

function createJobsNetwork_onet(svg, graph) {
    d3v4 = d3;

    let parentWidth = svg.node().parentNode.clientWidth;
    let parentHeight = svg.node().parentNode.clientHeight;

    // Define the div for the tooltip
    var div = d3.select("#d3_fd_network").append("div")	
        .attr("class", "tooltip")				
        .style("opacity", 0);


    // remove any previous graphs
    svg.selectAll('.g-main').remove();

    var gMain = svg.append('g')
    .classed('g-main', true);

    var rect = gMain.append('rect')
        .attr('width', parentWidth)
        .attr('height', parentHeight)
        .style('fill', 'white')

    var gDraw = gMain.append('g');

    var zoom = d3v4.zoom()
        .on('zoom', zoomed)

    gMain.call(zoom);

    function zoomed() {
        gDraw.attr('transform', d3v4.event.transform);
    }

    var color = d3v4.scaleOrdinal(d3v4.schemeCategory20);

    if (! ("links" in graph)) {
        console.log("Graph is missing links");
        return;
    }
    
    var i;
    for (i = 0; i < graph.nodes.length; i++) {
        graph.nodes[i].weight = 1.01;
    }
    var nodes = graph.nodes,
        nodeById = d3v4.map(nodes, function(d) { return d.id; }),
        links = graph.links,
        bilinks = [];
    //console.log(nodes)

    links.forEach(function(link) {
        var s = link.source = nodeById.get(link.source),
            t = link.target = nodeById.get(link.target);
        bilinks.push([s, t]);
    });

    var link = gDraw.append("g")
        .attr("class", "link")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .classed('node-links', true);
        //.attr("stroke-width", function(d) { return Math.sqrt(d.value); });

    var default_node_size_onet = 6,
        default_node_size_over = 8;
    

    var node = gDraw.append("g")
        .attr("class", "node")
        .selectAll("circle")
        .data(nodes.filter(function(d) { return d.id; }))
        .enter().append("g");

    node.append("circle")
        .classed('node', true)
        .attr("r", function(d) {
            if ('size' in d)
                return d.size;
            else
                return default_node_size_onet;
        })
        .attr("fill", function(d) {
            if ('color' in d)
                return d.color;
            else
                return color(d.group); 
        })
        .style("stroke", function(d) {
            if ('color' in d)
                return color(d.group);
            else
                return "#ffffff";
        })
        .style("opacity", function(d) {
            if('alpha' in d)
                return d.alpha
            else
                return 0.9;
        })
    //.on("click", click_node)

	.on("click",function(d,i){
        d3.select(this)
        .attr("r", function(d){return default_node_size_onet})
        .style("stroke", function(d) {return color(d.group)});

        let neighbors = links.filter(e => e.source.index === d.index || e.target.index === d.index).map(e => e.source.index === d.index ? e.target.index : e.source.index)

        d3.selectAll('circle.node')
            .style('opacity', e => { return e.index === d.index || neighbors.includes(e.index) ? 1 : 0.2})

        d3.selectAll('.node-links')
            .style('opacity', e => e.source.index === d.index || e.target.index === d.index ? 1 : 0.1)
            .style('stroke', e => e.source.index === d.index || e.target.index === d.index ? '#000' : '#f5f5f5')

        var selector = document.getElementById('community_div');
        selector.style.display="block";
        //d.group = 4;

        method_ngrams = document.getElementById('method-ngrams').value
        

        document.getElementById('cluster_group').value = ''
        document.getElementById('cluster_group').value = d.group
        generate_ngrams_list(d.group, method_ngrams);

        
        
        //nodeGrowing(d);
    })
        .on("mouseover", function(d,i) {
            var match_score = "";
            //if('value' in d) 
                //match_score = "<br/>" + d.weight + " %";
            
            d3v4.select(this)
                .transition()
                .attr("r", default_node_size_over);
                
            div.transition()
                .duration(500)		
                .style("opacity", .9);

            var matrix = this.getScreenCTM()
            .translate(+ this.getAttribute("cx"), + this.getAttribute("cy"));

            div.html("<br/>"  + d.tweet + "<br/>")
             .style("left", (window.pageXOffset + matrix.e-500) + "px")
             .style("top", (window.pageYOffset + matrix.f-500) + "px");
            //  .style("right", (d3.event.pageX-500) + "px")
            //  .style("top", (d3.event.pageY-500) + "px");

        })
        .on("mouseout", function(d,i) {
            if (!d3v4.select(this).classed("network-selected")) {
                d3v4.select(this)
                .transition()
                .duration(1000)
                .attr("r", function(d){
                    if ('size' in d)
                        return d.size;
                    else 
                        return default_node_size_onet;
                });
            }
            div.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .call(d3v4.drag());
    
    var simulation = d3v4.forceSimulation()
        .force("link", d3v4.forceLink()
        .id((d)=>{ return d.id; })
        .distance(function(d) { 
                    return 50;
                })
              )
        .force("charge", d3v4.forceManyBody().strength(-100).distanceMax(200))
        .force("center", d3v4.forceCenter(parentWidth / 2, parentHeight / 2))
        .force("x", d3v4.forceX())
        .force("y", d3v4.forceY().strength(function(d){ return .30 }));

    simulation
        .nodes(graph.nodes)
        .on("tick", ticked);

    simulation.force("link")
        .links(graph.links);

    function ticked() {
        // update node and line positions at every step of the force simulation
        link.attr("x1", function(d) { return d.source.x; })
            .attr("y1", function(d) { return d.source.y; })
            .attr("x2", function(d) { return d.target.x; })
            .attr("y2", function(d) { return d.target.y; });

        node.attr("transform", function(d) {
            return "translate(" + d.x + "," + d.y + ")";
        })
    }

    function click_node() {
        if (!d3v4.select(this).classed("network-selected")) {
            d3.selectAll('.network-selected').classed("network-selected", false)
                .transition().attr("r", function(d){
                    if ('size' in d)
                        return d.size;
                    else 
                        return default_node_size_onet;
                })
                .transition().style("stroke", (d) => color(d.group))
                .transition().style("stroke-width", 1.5);
            d3.selectAll('.network-selected').transition().attr("class", "noselected");
            
            d3.select(this).classed("network-selected", true)
            d3.select(this)
                .transition().style("stroke", "red")
                .transition().style("stroke-width", 2.5);

            //onClickParent(this.__data__.job_role)
        }
        else {
            d3.select(this).classed("network-selected", false);
            d3.select(this).transition().attr("class", "noselected");
        }
    }

    function nodeGrowing(d) {
        
        var nodeNeighbors = links.filter(function(link) { 
            // Filter the list of links to only those links that have our target
            // node as a source or target
            return link.source.index === d.index || link.target.index === d.index;})
        
        .map(function(link) {
            // Map the list of links to a simple array of the neighboring indices - this is
            // technically not required but makes the code below simpler because we can use
            // indexOf instead of iterating and searching ourselves.
            return link.source.index === d.index ? link.target.index : link.source.index; });

        // Reset all circles - we will do this in mouseout also
        //svg.selectAll('circle').style('stroke', 'pink');

        // now we select the neighboring circles and apply whatever style we want.
        // Note that we could also filter a selection of links in this way if we want to
        // Highlight those as well
        svg.selectAll('circle').filter(function(node) {
            // I filter the selection of all circles to only those that hold a node with an
            // index in my listg of neighbors
            return nodeNeighbors.indexOf(node.index) > -1;
        })
        .attr("r", 12)
        .transition()
        .duration(10);
    }

    
    return graph;
};

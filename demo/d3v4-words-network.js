function createWordsNetwork(svg, graph) {
    d3v4 = d3;

    let parentWidth = svg.node().parentNode.clientWidth;
    let parentHeight = svg.node().parentNode.clientHeight;

    

    // Define the div for the tooltip
    var div = d3.select("#network_words").append("div")	
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
        default_node_size_over = 10;

    graph.links.forEach(function(link){

        // initialize a new property on the node
        if (!link.source["linkCount"]) link.source["linkCount"] = 0; 
        if (!link.target["linkCount"]) link.target["linkCount"] = 0;
        
        // count it up
        link.source["linkCount"]++;
        link.target["linkCount"]++;    
        });

    let sizeScale = d3.scaleLinear()
        .domain(d3.extent(graph.nodes, d => d.linkCount))
        .range([5, 20])
    
    var node = gDraw.append("g")
        .attr("class", "node")
        .selectAll("circle")
        .data(nodes.filter(function(d) { return d.id; }))
        .attr("r", function(d){
            return d.linkCount ? (d.linkCount * 2) : 2; //<-- some function to determine radius
        })
        .enter().append("g");

    node.append("circle")
        //.attr("r", function(d) {
         //   if ('size' in d)
          //      return d.size;
          //  else
            //    return default_node_size_onet;
        //})
        .attr('r', d => sizeScale(d.linkCount))
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
        console.log(neighbors)
        d3.selectAll('circle.node')
            .style('opacity', e => { return e.index === d.index || neighbors.includes(e.index) ? 1 : 0.2})

        d3.selectAll('.node-links')
            .style('opacity', e => e.source.index === d.index || e.target.index === d.index ? 1 : 0.1)
            .style('stroke', e => e.source.index === d.index || e.target.index === d.index ? '#000' : '#f5f5f5')
    })
        .on("mouseover", function(d,i) {
            var match_score = "";
            //if('value' in d) 
                //match_score = "<br/>" + d.weight + " %";
            
            d3v4.select(this)
                .transition();
                
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
                .duration(1000);
            }
            div.transition()
                .duration(500)
                .style("opacity", 0);
        })
        .call(d3v4.drag());
    
    var simulation = d3v4.forceSimulation()
        .force("link", d3v4.forceLink()
                .id(function(d) { return d.id; })
                .distance(function(d) { 
                    return 15;
                    return dist; 
                })
              )
        .force("charge", d3v4.forceManyBody())
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
    return graph;
};

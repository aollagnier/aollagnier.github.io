function createJobsNetwork_dwp(svg, graph) {
    d3v4 = d3;

    let parentWidth = d3v4.select('svg').node().parentNode.clientWidth;
    let parentHeight = d3v4.select('svg').node().parentNode.clientHeight;

    var svg = d3v4.select('svg')
        .attr('width', parentWidth)
        .attr('height', parentHeight)

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
        .data(graph.links)
        .enter().append("line")
        .attr("stroke-width", function(d) { return Math.sqrt(d.value); });

    var default_node_size_dwp = 6,
        default_node_size_over = 10;    
    
    var node = gDraw.append("g")
        .attr("class", "node")
        .selectAll("circle")
        .data(graph.nodes)
        .enter().append("g");

    node.append("circle")
        .attr("r", function(d){
            if ('size' in d)
                return d.size;
            else 
                return default_node_size_dwp;
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
                return 1;
        })
        .on("click", click_node)
        .on("mouseover", function(d,i) {
            var match_score = "";
            if('Role_Match_Score' in d) 
                match_score = "<br/>" + d.Role_Match_Score + " %";
            
            d3v4.select(this)
                .transition()
                .attr("r", default_node_size_over);
                
            div.transition()
                .duration(500)		
                .style("opacity", .9);

            div.html("<br/>"  + d.job_role + match_score + "<br/>")	
                .style("left", (d3v4.event.pageX + 15) + "px")		
                .style("top", (d3v4.event.pageY - 10) + "px");
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
                            return default_node_size_dwp;
                    });
            }
            div.transition()
                .duration(500)
                .style("opacity", 0)
        })
        .call(d3v4.drag());
    
    node.append("text")
        .text(function(d) {
            return d.job_role;
        })
        .attr('x', function (d) {
            var label = d.label_pos;
            console.log(d.job_role.length, label)
            if (label.localeCompare("left") == 0)
                return - (d.job_role.length * 5) - 10;
            else
                return 10;           
        })
        .attr('y', function (d) {
            var label = d.label_pos;
            if (label.localeCompare("top") == 0 || label.localeCompare("left") == 0)
                return 0;     
            else
                return 10;
        });
       
    var simulation = d3v4.forceSimulation()
        .force("link", d3v4.forceLink()
                .id(function(d) { return d.id; })
                .distance(function(d) { 
                    return 30;
                })
              )
        .force("charge", d3v4.forceManyBody())
        .force("center", d3v4.forceCenter(parentWidth / 2, parentHeight / 2));
        //.force("x", d3v4.forceX(parentWidth/2))
        //.force("y", d3v4.forceY(parentHeight/2));

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
            
            d3.selectAll('.network-selected')
                .classed("network-selected", false)
                .transition().attr("r", function(d){
                    if ('size' in d)
                        return d.size;
                    else 
                        return default_node_size_dwp;
                })
                .transition().style("stroke", (d) => color(d.group))
                .transition().style("stroke-width", 1.5);
            d3.selectAll('.network-selected').transition().attr("class", "noselected");
            
            d3.select(this).classed("network-selected", true)
            d3.select(this)
                .transition().style("stroke", "red")
                .transition().attr("stroke-width", 2.5);
            //onClickParent(this.__data__.job_role)
        }
        else {
            d3.select(this).classed("network-selected", false);
            d3.select(this).transition().attr("class", "noselected");
        }
    }

    var legend_w = 250, legend_h = 150; 
    gDraw.selectAll('image').data([0])
        .enter()
        .append("svg:image")
        .attr('xlink:href', 'data/network_legend.png')
        .attr("x", function(d) {return parentWidth - (legend_w + 30)})
        .attr("y", function(d) {return parentHeight - (legend_h + 20)})
        .attr("width", legend_w)
        .attr("height", legend_h);

    var texts = ['+ Use the scroll wheel to zoom',
                 '+ Hold the mouse over a node to display the job role name'];

    gDraw.selectAll('text')
        .data(texts)
        .enter()
        .append('text')
        .attr('x', 10)
        .attr('y', function(d,i) { return 400 + i * 20; })
        .text(function(d) { return d; });
    return graph;
};
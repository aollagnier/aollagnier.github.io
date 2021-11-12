function get_jobs_network(graph, user_profile, k_nodes_display) {
    // console.log(graph);
    
    if (k_nodes_display === undefined) {
        k_nodes_display = Math.min(10, user_profile.length);
    }
    k_nodes_display = Math.min(k_nodes_display, user_profile.length);

    bounds = get_bound_scores(user_profile, k_nodes_display);
    // console.log(bounds)
    var i, j, i_node = 0;
    for (i = 0; i < graph.nodes.length; i++) {
        graph.nodes[i].Role_Match_Score = 0;
        graph.nodes[i].size = 4;
        graph.nodes[i].color = "#fff";
        graph.nodes[i].alpha = 0.8;
        graph.nodes[i].group = graph.nodes[i].group;

        job_role = graph.nodes[i].job_role;

        for (j = 0; j < user_profile.length; j++) {
            if (job_role.localeCompare(user_profile[j].Role) == 0) {
                role_score = user_profile[j].Role_Match_Score;
                graph.nodes[i].Role_Match_Score = role_score
                
                if (i_node < k_nodes_display && role_score >= bounds[0] && role_score <= bounds[1]) {
                    
                    graph.nodes[i].size     = get_node_size(role_score, bounds[0], bounds[1]); 
                    graph.nodes[i].color    = get_node_color(role_score, bounds[0], bounds[1]);
                    graph.nodes[i].alpha    = 0.9;
                    i_node = i_node + 1;
                    // console.log([job_role,  role_score])
                }
                break;
            }
        }
    }
    // console.log(graph)
    return graph;
}
function get_node_color(x, mnx, mxx) {
    colormap = ['#00224D', '#00234F', '#002350', '#002452', '#002554', '#002655', '#002657', '#002759', '#00285B', '#00285C', '#00295E', '#002A60', '#002A62', '#002B64', '#002C66', '#002C67', '#002D69', '#002E6B', '#002F6D', '#002F6F', '#003070', '#003070', '#003170', '#003170', '#043270', '#083370', '#0B3370', '#0E3470', '#11356F', '#14366F', '#16366F', '#18376F', '#1A386F', '#1C386E', '#1D396E', '#1F3A6E', '#223B6E', '#243C6E', '#253D6D', '#273D6D', '#283E6D', '#2A3F6D', '#2B3F6D', '#2C406D', '#2E416C', '#2F426C', '#30426C', '#31436C', '#32446C', '#34446C', '#35456C', '#36466C', '#37466C', '#38476C', '#39486C', '#3A486B', '#3B496B', '#3D4A6B', '#3E4B6B', '#3F4B6B', '#404C6B', '#414D6B', '#424D6B', '#434E6B', '#444F6B', '#454F6B', '#46506B', '#47516B', '#48516B', '#49526B', '#4A536B', '#4B546C', '#4D556C', '#4E566C', '#4E566C', '#4F576C', '#50586C', '#51586C', '#52596C', '#535A6C', '#545A6C', '#555B6D', '#565C6D', '#575D6D', '#585D6D', '#595E6D', '#595F6D', '#5A5F6D', '#5B606E', '#5C616E', '#5D616E', '#5E626E', '#5F636E', '#60646E', '#61646F', '#61656F', '#62666F', '#63666F', '#64676F', '#656870', '#666970', '#676970', '#686A70', '#686B71', '#696B71', '#6A6C71', '#6B6D71', '#6D6E72', '#6E6F72', '#6E7073', '#6F7073', '#707173', '#717273', '#727374', '#737374', '#747475', '#747575', '#757575', '#767676', '#777776', '#787876', '#797877', '#797977', '#7A7A77', '#7B7B77', '#7C7B78', '#7D7C78', '#7E7D78', '#7F7D78', '#807E78', '#817F78', '#828078', '#838078', '#848178', '#858278', '#858378', '#868378', '#878478', '#888578', '#898678', '#8A8678', '#8B8778', '#8C8878', '#8E8978', '#8F8A77', '#908B77', '#918C77', '#928C77', '#938D77', '#948E77', '#958F77', '#968F77', '#979076', '#989176', '#999276', '#9A9376', '#9B9376', '#9C9476', '#9D9575', '#9E9675', '#9F9675', '#A09775', '#A19874', '#A29974', '#A39A74', '#A49A74', '#A59B73', '#A69C73', '#A79D73', '#A89E73', '#A99E72', '#AA9F72', '#ABA072', '#ACA171', '#ADA271', '#AEA271', '#AFA370', '#B0A470', '#B2A66F', '#B3A66F', '#B4A76F', '#B5A86E', '#B6A96E', '#B7AA6D', '#B8AB6D', '#B9AB6D', '#BAAC6C', '#BBAD6C', '#BCAE6B', '#BDAF6B', '#BEB06A', '#BFB06A', '#C1B169', '#C2B269', '#C3B368', '#C4B468', '#C5B567', '#C6B567', '#C7B666', '#C8B765', '#C9B865', '#CAB964', '#CBBA64', '#CCBB63', '#CDBC62', '#CEBC62', '#CFBD61', '#D0BE60', '#D2BF60', '#D3C05F', '#D4C15E', '#D5C25E', '#D6C35D', '#D7C35C', '#D9C55A', '#DAC65A', '#DBC759', '#DCC858', '#DEC957', '#DFCA56', '#E0CB55', '#E1CC54', '#E2CC53', '#E3CD52', '#E4CE51', '#E5CF50', '#E6D04F', '#E8D14E', '#E9D24D', '#EAD34C', '#EBD44B', '#ECD54A', '#EDD648', '#EED747', '#EFD846', '#F1D944', '#F2DA43', '#F3DA42', '#F4DB40', '#F5DC3F', '#F6DD3D', '#F8DE3B', '#F9DF3A', '#FAE038', '#FBE136', '#FDE234', '#FDE333', '#FDE534', '#FDE636', '#FDE737'];
    
    a = 0; b = colormap.length;
    pos = (b-a)*((x-mnx)/(mxx-mnx)) + a;
    
    pos = b - Math.floor(pos);
    if (pos == b)
        pos = b-1
    // console.log([x, pos, b])
    return colormap[pos];
}

function get_node_size(x, mnx, mxx){
    a = 5; b = 10;
    nsize = (b-a)*((x-mnx)/(mxx-mnx)) + a;
    // console.log([x, mnx, mxx])
    // console.log(nsize)
    return nsize

}

function get_bound_scores(user_profile, n_scores) {
    scores = [];
    for (i = 0; i < n_scores; i++) {
        scores[i] = user_profile[i].Role_Match_Score;
    }
    // console.log([user_profile, n_scores])
    // console.log(scores)
    return [Math.min.apply(Math, scores), Math.max.apply(Math, scores)]
}

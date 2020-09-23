// let curData = [];


load_data_light().then(r => init(r));


async function load_data_light() {


    return await d3.json('static/assets/data/data.json', d3.autoType)
}


function init(data) {

    //
    // let form = new FormData();
    // form.append("units", []);
    //
    //
    // $.ajax({
    //     type: "POST",
    //     url: "/proj",
    //     processData: false,
    //     contentType: false,
    //     data: form,
    //     success: function (d) {
    //         if (d !== "over") {
    //
    //             console.log(d);

    console.log(data);

    // data = JSON.parse(data)

    plotter_init(data);
    fillSelect(data.map(d => d.global_group), "#ggroup")
    fillSelect2(data.map(d => d.functions), "#function")


    //do STUFF
    //         }
    //     }
    // });
}


function fillSelect(data, id) {

    let map = new Map();
    let sel = $(id);
    // let select = document.getElementById('ggroup');

    console.log(sel);

    for (let i = 0; i < data.length; i++) {
        map.set(data[i], "bob")
    }


    let temp = Array.from(map.keys());
    console.log(temp);

    let mes = ""
    for (let i = 0; i < temp.length; i++) {


        sel.append(new Option(temp[i], temp[i]))
        sel.append();
    }


}


function fillSelect2(data, id) {

    let map = new Map();
    let sel = $(id);
    // let select = document.getElementById('ggroup');

    console.log(sel);

    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            map.set(data[i][j], "bob")
        }

    }


    let temp = Array.from(map.keys());
    console.log(temp);

    let mes = ""
    for (let i = 0; i < temp.length; i++) {


        sel.append(new Option(temp[i], temp[i]))
        sel.append();
    }


}


function plotter_init(data) {


    let svg = d3.select("#proj")

    let svg1 = document.getElementById('proj');

    let bBox = svg1.getBBox();


    let xrange = d3.extent(data.map(d => d['k_dist'][0]));
    let yrange = d3.extent(data.map(d => d['k_dist'][1]));
    console.log(bBox.width);

    let xscale = d3.scaleLinear().domain(xrange).range([0, 960]);
    let yscale = d3.scaleLinear().domain(yrange).range([0, 642]);


    svg.selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
        .attr("cx", d => {
            return xscale(d['k_dist'][0])
        })
        .attr("cy", d => yscale(d['k_dist'][1]))
        .attr("r", 2.5)
        .attr("stroke", "#555555")
        .style("stroke-width", "0.1")
        .style("opacity", "0.8")
        .attr("fill", d => d.reasoning_label === "bias" ? "red" : "green")


}

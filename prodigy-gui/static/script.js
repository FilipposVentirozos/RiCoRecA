require('vis-network');

document.addEventListener('prodigyupdate', event => {
vis_render();
})
// Documentation for drawing
// https://visjs.github.io/vis-network/docs/network/nodes.html
// https://visjs.github.io/vis-network/docs/network/edges.html
function vis_render(){
  const eg = window.prodigy.content
  // create an array with nodes
  let node_list = [];
    for (let i = 0; i < eg.spans.length; i++) {
      switch (eg.spans[i].label ) {
        case "ACTION":
          node_list.push({id: eg.spans[i].token_end, value: 18, color: "#E15233",
              label: (eg.text.substring(eg.spans[i].start, eg.spans[i].end).length > 16) ?
                  (eg.text.substring(eg.spans[i].start, eg.spans[i].end).substring(0,16) + "...")
                  : (eg.text.substring(eg.spans[i].start, eg.spans[i].end)),
              title: "Action", shape: "ellipse",
              scaling: {label: {enabled: true, min: 20, max: 20}}} ); // Size for nodes with labels inside them
          break;
          case "INGR":
          node_list.push({id: eg.spans[i].token_end, value: 10, color: "#EFAE2C", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), font: {size: 10, ital: true} } );
          break;
        case "TOOL":
          node_list.push({id: eg.spans[i].token_end, value: 12, color: "#12A5E9", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end)} );
          break;
        case "STT_INGR":
          node_list.push({id: eg.spans[i].token_end, value: 10, color: "#EFAE2C", label: "State\nIngr.",
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "triangle"} );
          break;
        case "STT_TOOL":
          node_list.push({id: eg.spans[i].token_end, value: 12, color: "#12A5E9", label: "State\nTool",
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "triangle"} );
          break;
        case "COR_INGR":
          node_list.push({id: eg.spans[i].token_end, value: 10, color: "#EFAE2C", label: "Coref.\nIngr.",
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "hexagon"} );
          break;
        case "COR_TOOL":
          node_list.push({id: eg.spans[i].token_end, value: 12, color: "#12A5E9", label: "Coref.\nTool",
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "hexagon"} );
          break;
        case "PAR_INGR":
          node_list.push({id: eg.spans[i].token_end, value: 11, color: "#f3bf59", label: "Part\nIngr.",
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "square"} );
          break;
        case "PAR_TOOL":
          node_list.push({id: eg.spans[i].token_end, value: 12, color: "#2ab0ef", label: "Part\nTool",
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "square"} );
          break;
        case "HOW":
        case "H":
        case "VALUE_H":
        case "UNIT_H":
          node_list.push({id: eg.spans[i].token_end, value: 13, color: "#BBE43C", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "box"} );
          break;
        case "MSR":
          node_list.push({id: eg.spans[i].token_end, value: 13, color: "#a99670", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "box"} );
          break;
        case "SETT":
          node_list.push({id: eg.spans[i].token_end, value: 13, color: "#5e889c", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "box"} );
          break;
        case "WHY":
          node_list.push({id: eg.spans[i].token_end, value: 10, color: "#5804b3", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "star"} );
          break;
        default:
          node_list.push({id: eg.spans[i].token_end, value: 13, color: "#82A5B6", label: eg.spans[i].label,
              title:eg.text.substring(eg.spans[i].start, eg.spans[i].end), shape: "diamond"} );
          break;
      }
    }
  let nodes = new vis.DataSet(node_list);
  let edge_list = [];
  for (let i = 0; i < eg.relations.length; i++) { // why here
      switch (eg.relations[i].label ) {
        case "Member":
            edge_list.push({from: eg.relations[i].child, to: eg.relations[i].head, title:eg.relations[i].label,
                width:7, color:"#62C7B5",
                arrows: {from: {enabled: true}}}) // For no direction
            break;
        case "Dependency":
            edge_list.push({from: eg.relations[i].child, to: eg.relations[i].head, title:eg.relations[i].label,
                arrows: "to", width:8, color:"#E38527", })
            break;
        case "Modifier":
            edge_list.push({from: eg.relations[i].child, to: eg.relations[i].head, title:eg.relations[i].label,
                arrows: "from", width:3, color:"#65C762", label: "Mod", font: {background: "#65C762"}})
            break;
        default:
            edge_list.push({from: eg.relations[i].child, to: eg.relations[i].head, // title:eg.relations[i].label,
                arrows: {to: {enabled: false}},
                width:3, color:"#82A5B6", label: eg.relations[i].label, font: {background: "#82A5B6"}})
            break;
      }
    }
  // create an array with edges
  let edges = new vis.DataSet(edge_list);
  // create a network
  let container = document.getElementById("mynetwork");
  let data = {
      nodes: nodes,
      edges: edges
    };
    let options = {
      nodes: {
        shape: "dot"
      }
    };
    var network = new vis.Network(container, data, options);
   // Do something with the annotations here
   // console.log('Updated', eg)
   // console.log('Updated', eg.spans)
   // console.log('Updated', eg.spans[0])
    console.log('Updated', eg)
}

// const Labels = ["INGR", "PAR_INGR", "TOOL", "PAR_TOOL"];
const Labels = ["INGR", "TOOL"];
const Hs = ["HOW"]  // , "VALUE_H", "UNIT_H"];

function add_Hs(eg, H_tuple, token_end_){
    // console.log("inside add_Hs() function")
    // console.log(H_tuple, token_end_)
    for (let j = 0; j < eg.relations.length; j++){
        if (eg.relations[j].child === token_end_){ // Check if selected token is in relation
            let idx = Hs.indexOf(eg.relations[j].head_span.label); // Check if it's relation is in Hs
            if (idx >= 0){
                // console.log("We are inside!");
                if (!H_tuple.hasOwnProperty(idx)){ // Check if already exists
                    H_tuple[idx] = eg.text.substring(
                                eg.relations[j].head_span.start,
                                eg.relations[j].head_span.end);
                    // console.log("Token after insertion inside!");
                    H_tuple = add_Hs(eg, H_tuple, eg.relations[j].head_span.token_end);
                }
            }
        }else if(eg.relations[j].head === token_end_){
            let idx = Hs.indexOf(eg.relations[j].child_span.label);
            if (idx >= 0){
                // console.log("We are inside!");
                if (!H_tuple.hasOwnProperty(idx)){ // Check if already exists
                    H_tuple[idx] = eg.text.substring(
                                eg.relations[j].child_span.start,
                                eg.relations[j].child_span.end);
                    // console.log("Token after insertion inside!")
                    H_tuple = add_Hs(eg, H_tuple, eg.relations[j].child_span.token_end);
                }
            }
        }
    }
    return H_tuple;
}


function check_link_verb(eg, Action_token, Entity_token){
    // console.log("inside add_Hs() function")
    // console.log(H_tuple, token_end_)
    for (let j = 0; j < eg.relations.length; j++){
        if (eg.relations[j].child === Action_token){ // Check if selected token is in relation
            if (eg.relations[j].head === Entity_token){
                return true
            }
        }else if(eg.relations[j].head === Action_token){
            if (eg.relations[j].child === Entity_token){
                return true
            }
        }
    }
    return false;
}


function in_sent(eg, t_e){
    counter = 0
    for (let i = 0; i < eg.meta.sentences.length; i++){
        counter += eg.meta.sentences[i].length + 1; // The 1 is added for the space
        // console.log(counter)
        // console.log(t_e)
        if (t_e < counter){
            return i
        }
    }
}

function defaultDict(map, defaultd) {
    return function(key) {
        if (key in map)
            return map[key];
        if (typeof defaultd == "function")
            return defaultd(key);
        return defaultd;
    };
}

function download_csv() {
    const eg = window.prodigy.content;
    // let entities = [{},{}]; // Schema: list of span type ("INGR", "TOOL", then
    // dictionaries of [span, sentence_id]

    // Set the ingredients uniquely and tools from the annotation. Token spans come in order.
    // let sentence_idx = 0;
    // Find the entities
    let ingrs = {};
    let tools = {};
    for (let i = 0; i < eg.spans.length; i++) {
        // "INGR",  "TOOL",
        if (eg.spans[i].label === "INGR"){
            if (eg.text.substring(eg.spans[i].start, eg.spans[i].end) in ingrs){
                ingrs[eg.text.substring(eg.spans[i].start, eg.spans[i].end)].push(eg.spans[i].token_end);
            }else{
                ingrs[eg.text.substring(eg.spans[i].start, eg.spans[i].end)] = [eg.spans[i].token_end];
            }

        }else if(eg.spans[i].label === "TOOL"){
            if (eg.text.substring(eg.spans[i].start, eg.spans[i].end) in ingrs){
                tools[eg.text.substring(eg.spans[i].start, eg.spans[i].end)].push(eg.spans[i].token_end);
            }else{
                tools[eg.text.substring(eg.spans[i].start, eg.spans[i].end)] = [eg.spans[i].token_end];
            }
        }
    }
    let sents_dict = []; // Schema: list with the Action words as index then the sentences were they belong etc.
    let action_entities = [];
    for (let i = 0; i < eg.spans.length; i++) {
        // ACTION
        if (eg.spans[i].label === "ACTION"){
            // Start from unseen Actions.
            let total_chars =0
            for (let j = 0; j < eg.meta.sentences.length; j++) {
                total_chars += eg.meta.sentences[j].length;
                if (eg.spans[i].start > total_chars) {
                    continue;
                }
                let tmp = [];
                tmp.push(eg.text.substring(eg.spans[i].start, eg.spans[i].end));
                tmp.push(eg.meta.sentences[j]);
                tmp.push(j); // Sent_id
                // Token_id. Get the index of the Action word, after a possible existent in the same sentence.
                // tmp.push(eg.meta.sentences[j].indexOf(eg.text.substring(eg.spans[i].start, eg.spans[i].end),
                //     token_id) + 1);
                // and spans to be recognisable
                tmp.push(eg.spans[i].start);
                tmp.push(eg.spans[i].end);
                //  Add Token end for relation extraction downstream purposes
                tmp.push(eg.spans[i].token_end);
                sents_dict.push(tmp);
                // console.log(tmp);
                break;
            }
            // Find its entities
            let token_end = eg.spans[i].token_end; // Identify the ID token of the span
            let tmp = [];
            for (let j = 0; j < eg.relations.length; j++){
                if (eg.relations[j].child === token_end ){
                    if (eg.relations[j].head_span.label === "INGR" || eg.relations[j].head_span.label === "TOOL"){
                        tmp.push(eg.relations[j].head);
                    }
                }else if (eg.relations[j].head === token_end){
                    if (eg.relations[j].child_span.label === "INGR" || eg.relations[j].child_span.label === "TOOL"){
                        tmp.push(eg.relations[j].child);
                    }
                }
            }
            action_entities.push(tmp);
        }
    }
    console.log(action_entities)
    console.log(ingrs)
    console.log(tools)
    //Debug
    // console.log(sents_dict);
    // console.log(entities);
    // Construct the first and second header depending on the options
    let ex_header = ["", ""];
    let header = ["Actions", "Sentences"];
    for (let [key, _] of Object.entries(ingrs)) {
       ex_header.push("INGR");
        header.push(key);
    }
    for (let [key, _] of Object.entries(tools)) {
       ex_header.push("TOOL");
        header.push(key);
    }
    header.push("");
    header.push("Notes");
    header.push("");
    header.push("Action_start");
    header.push("Action_end");
    header.push("Sentence_Id");
    ex_header.push("");
    ex_header.push("");
    ex_header.push("");
    ex_header.push("Metadata");
    ex_header.push("Metadata");
    ex_header.push("Metadata");
    // console.log(entities_header)
    // console.log(header)
    let data = [];
    // data.push(ex_header);
    // data.push(header);
    for (let i = 0; i < sents_dict.length; i++) { // Essentially row iteration
        let tmp = [];
        tmp.push(sents_dict[i][0]);
        tmp.push(sents_dict[i][1]);
        // for (let j = 0; j < entities_header.length; j++){
        for (let [_ , value] of Object.entries(ingrs)) {
            let cell = [];
            for (let token_id of value){
                if (action_entities[i].includes(Number(token_id))) {
                    cell = [1];
                    tmp.push(cell);
                    console.log("Hurray!")
                }
            }
            if (cell.length === 0){
                tmp.push("");
            }
        }
        for (let [_, value] of Object.entries(tools)) {
            let cell = [];
            for (let token_id of value){
                if (action_entities[i].includes(Number(token_id))) {
                    cell = [1];
                    tmp.push(cell);
                }
            }
            if (cell.length === 0){
                tmp.push("");
            }
        }
        tmp.push("");
        tmp.push("");
        tmp.push("");
        tmp.push(sents_dict[i][3]);
        tmp.push(sents_dict[i][4]);
        tmp.push(sents_dict[i][2]);
        // tmp.unshift(sents_dict[i][0], sents_dict[i][1]);
        // tmp.splice(-1,0, sents_dict[i][4])
        // tmp.splice(-1 ,1, sents_dict[i][5])
        // console.log(tmp)
        data.push(tmp);
    }
    // The duplicate row at the end for the annotators
    let dupl_row = ["Duplicate of: "];
    dupl_row.push.apply(dupl_row, Array(header.length - 1 ));
    data.push(dupl_row);
    // The User added entity row
    let anno_added = ["Annotator added: "];
    anno_added.push.apply(anno_added, Array(header.length - 1 ));
    data.push(anno_added);

    // console.log("The Dataset")
    // console.log(data);
    header = header.join("|").concat("\n");
    // console.log(header)
    // console.log(ex_header)
    // console.log(data)
    data.forEach(function(row) { // It appends the data to the Header
            // ex_header += row.join('|');  // use pipe to separate values
            // ex_header += "\n";
            header += row.join('|');  // use pipe to separate values
            header += "\n";
    });
    // console.log(header)
    ex_header = ex_header.join("|").concat("\n");
    ex_header += header
    let hiddenElement = document.createElement('a');
    hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(ex_header);
    hiddenElement.target = '_blank';
    hiddenElement.download = eg.meta.id.concat('.psv');
    hiddenElement.click();
}

//
// // XLSX = require('/home/chroner/PhD_remote/FoodBase/static/xlsx.full.min.js');
// // XLSX = require('xlsx.full.min.js');
// // import "./xlsx.full.min"
//
// function doit(type, fn, dl) {
// 	var elt = document.getElementById('data-table');
// 	var wb = XLSX.utils.table_to_book(elt, {sheet:"Sheet JS"});
// 	return dl ?
// 		XLSX.write(wb, {bookType:type, bookSST:true, type: 'base64'}) :
// 		XLSX.writeFile(wb, fn || ('SheetJSTableExport.' + (type || 'xlsx')));
// }
//
// function tableau(pid, iid, fmt, ofile) {
// 	if(typeof Downloadify !== 'undefined') Downloadify.create(pid,{
// 			swf: 'downloadify.swf',
// 			downloadImage: 'download.png',
// 			width: 100,
// 			height: 30,
// 			filename: ofile, data: function() { return doit(fmt, ofile, true); },
// 			transparent: false,
// 			append: false,
// 			dataType: 'base64',
// 			onComplete: function(){ alert('Your File Has Been Saved!'); },
// 			onCancel: function(){ alert('You have cancelled the saving of this file.'); },
// 			onError: function(){ alert('You must put something in the File Contents or there will be nothing to save!'); }
// 	}); else document.getElementById(pid).innerHTML = "";
// }
//
// /* initial table */
// var aoa = [
// 	["This",   "is",     "a",    "Test"],
// 	["வணக்கம்", "สวัสดี", "你好", "가지마"],
// 	[1,        2,        3,      4],
// 	["Click",  "to",     "edit", "cells"]
// ];
// var ws = XLSX.utils.aoa_to_sheet(aoa);
// var html_string = XLSX.utils.sheet_to_html(ws, { id: "data-table", editable: true });
// document.getElementById("container").innerHTML = html_string;
// tableau('odsbtn',   'xportods',   'ods',   'SheetJSTableExport.ods');
// tableau('fodsbtn',  'xportfods',  'fods',  'SheetJSTableExport.fods');
// tableau('xlsxbtn',  'xportxlsx',  'xlsx',  'SheetJSTableExport.xlsx');
//
//
//

import React, {useEffect, useState, useCallback} from 'react';
import {TextField, Button} from '@mui/material';
import Box from '@mui/material/Box';

import './App.css';

// https://stackoverflow.com/questions/72022246/how-to-pass-a-flask-post-to-react
// https://stackoverflow.com/questions/55757761/handle-an-input-with-react-hooks

function App() {

    const [query, setQuery] = useState('')
    // const [inputValue, setInputValue] = React.useState("");

    const onChangeHandler = event => {
        setQuery(event.target.value);
     };

    const [backendData, setBackendData] = useState([{}]);

    const [recName, setRecName] = useState("");
    const [recImg, setRecImg] = useState("");

    // useEffect(() => { https://stackoverflow.com/questions/56247433/how-to-use-setstate-callback-on-react-hooks
    // https://stackoverflow.com/questions/72113386/reactjs-how-do-you-call-an-async-fuction-on-button-click
    // https://stackoverflow.com/questions/72113386/reactjs-how-do-you-call-an-async-fuction-on-button-click
    // https://stackoverflow.com/questions/71174509/proper-async-await-syntax-for-fetching-data-using-useeffect-hook-in-react
    // https://stackoverflow.com/questions/56663785/invalid-hook-call-hooks-can-only-be-called-inside-of-the-body-of-a-function-com

     // https://stackoverflow.com/questions/40825010/const-or-let-in-react-component
    let handleTransfer = useCallback( async () => {

    // async function handleTransferHelper() {
        fetch('/query', {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({"query_body": query})
        }).then(() => {
            console.log("POST: " + query)
        })
          
        // https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
        // https://stackoverflow.com/questions/54936559/using-async-await-in-react-component
        const response = await fetch("/query"); 
        const games = await response.json();
            console.log(games);
             setRecName(games.game)
             setRecImg(games.url)

            
            console.log(recName)
            console.log(recImg)

            console.log(games['game'])
            console.log(games.game)
            console.log(recName)
            console.log(recImg)
    })
    // handleTransferHelper()
    // }, []);

    // https://stackoverflow.com/questions/51184136/display-an-image-from-url-in-reactjs

    // https://stackoverflow.com/questions/46966413/how-to-style-material-ui-textfield

    //.MuiTextField-root {
  //align-items: center;
  //text-align: center;
//}

    return (
    <div className="container">
        <h1>Video Game Recommender</h1>
    
        <Box
        // box for textfield and button
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="5vh">
        
            <TextField 
            // textfield inside box to center component
            // https://stackoverflow.com/questions/56500234/how-can-i-center-align-material-ui-textfield-text-and-also-set-a-min-number-valu

            InputProps={{
                inputProps: {
                    style: { textAlign: "center", alignItems: 'center', justifyContent: 'center'},  
                }
            }}
            id="video_game_entry" 
            label="Search" 
            variant="outlined"
            placeholder="video game keywords"
            helperText="Please enter some keywords to get recommended video games."
            type="text"
            name="name"
            onChange={onChangeHandler}
            value={query}
            alignItems='center'/>

            <Button variant="text" onClick={handleTransfer}> Transfer Data </Button>
        </Box>

        <Box
        display="flex"
        justifyContent="center"
        alignItems="center">
        <h3>{backendData.query_body}</h3>
            <h3>{recName}</h3>
        </Box>
        
        <Box
        display="flex"
        justifyContent="center"
        alignItems="center" // vh: 5% of viewport height
        
        // <h3>{recName}</h3>
        >

            
            <img 
            src={recImg}
            alt=""
            height="50%"
            width="50%"
            />
        </Box>
        
        </div>

    );


}

            //useEffect(() => { setRecName(games.game) }, [])
            //useEffect(() => { setRecImg(games.url) }, [])
  
        // fetch('/query').then(
          // response => response.json()
        // ).then(data => setBackendData(data)).then((data) => {
           // console.log("GET: " + data)
        // })

    

export default App;

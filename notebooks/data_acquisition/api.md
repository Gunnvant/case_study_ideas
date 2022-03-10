## Documentation for the api service created.

Only one endpoint has been created which is `POST /russia`. 
Body of this post will be:
```python
{
    "category":"Tanks"
}

```

Sample response will be 
```python
{
    "category": "Tanks",
    "details": {
        "abandoned": 30.0,
        "captured": 71.0,
        "damaged": 2.0,
        "destroyed": 53.0,
        "total": 156.0
    }
}

```

Following is the list of valid category names:

```shell
    Tanks
    Armoured Fighting Vehicles 
    Infantry Fighting Vehicles
    Armoured Personnel Carriers
    Mine Resistant Ambush Protected MRAP Vehicles
    Infantry Mobility Vehicles
    Communications Stations
    Engineering Vehicles
    Anti Tank Guided Missiles
    Man Portable Air Defence Systems
    Heavy Mortars
    Towed Artillery
    Self Propelled Artillery
    Multiple Rocket Launchers
    Self propelled Anti Aircraft Guns
    Surface To Air Missile Systems
    Radars
    Jammers And Deception Systems
    Aircraft
    Unmanned Aerial Vehicles
    Helicopters
    Logistics Trains

```
If wrong category is sent as a payload then a `400` error is raised.
function FindProxyForURL(url, host){
	if (shExpMatch(host, "*.dmm.com")) {
		return "DIRECT";
	} else if (shExpMatch(host, "*.google.com")) {
		return "DIRECT";
	} else {
		return "SOCKS localhost:1080; DIRECT";
	}
}

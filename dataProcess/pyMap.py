import os, shlex, subprocess, unicodedata, string

class MetaMap():


    def __init__(self):
        self.skrmedpost_server = None


    def start_server(self, server):
        '''
        Create a MetaMap subprocess based on the supplied server command.

        Arguments:
            server (str): The MetaMap start file location. It must be the
                          absolute path if the directory is not in $PATH.

        Returns:
            proc: A subprocess corresponding to the supplied server location.

        '''
        proc = subprocess.Popen(server,
                                stdout=subprocess.PIPE)
        return proc


    def start_skrmedpost_server(self, server_loc=None):
        '''
        Start the MetaMap skrmedpost server.

        Arguments:
            server_loc (str): The MetaMap start file location. It must be the
                               absolute path if the directory is not in $PATH.

        '''
        if server_loc:
            server = server_loc
        else:
            server = 'skrmedpostctl_start.bat'
        self.skrmedpost_server = self.start_server(server)


    def stop_skrmedpost_server(self):
        '''
        Terminate the skrmedpost server process.

        '''
        self.skrmedpost_server.terminate()


    def start_wsd_server(self, server_loc=None):
        '''
        Start the MetaMap word sense disambiguation server.

        Arguments:
            server_loc (str): The MetaMap start file location. It must be the
                               absolute path if the directory is not in $PATH.

        '''
        if server_loc:
            server = server_loc
        else:
            server = 'wsdserverctl_start.bat'
        self.wsd_server = self.start_server(server)


    def stop_wsd_server(self, server_loc=None):
        '''
        Terminate the wsd server process.
        
        '''
        self.wsd_server.terminate()


    def process_text(self, text, MM_options='metamap14.bat --XMLf'):
        '''
        Submit a block of text to MetaMap for processing. This requires
        that the skrmedpost server is running. 

        Arguments:
            text (str): The text to be processed. 

            MM_options (str): The same command you would run from the command
                               line to start the MetaMap process. 

        Returns:
            out (str): The results of processing the text as returned by the
                       MetaMap process.

        '''
        text = self.sanitize_text(text)
        p = subprocess.Popen(shlex.split(MM_options), stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE)
        out = p.communicate(input=text+'\n')[0]
        p.terminate()
        return out


    def sanitize_text(self, text):
        '''
        MetaMap only accepts ASCII printable characters. This function will
        ensure that extraneous characters are replaced or removed before 
        submitting to the MetaMap process.

        Arguments:
            text (str): The raw, unprocessed text.

        Returns:
            text (str): A pure-ASCII, sanitized text.

        '''
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(list(filter(lambda x: x in string.printable, text)))
        return text

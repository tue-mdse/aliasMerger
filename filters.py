"""
	Copyright 2012-2013
	Eindhoven University of Technology (Erik Kouters, Bogdan Vasilescu, Alexander Serebrenik)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import itertools
import re
import unicodedata
# Not part of standard library:
from dictUtils import MyDict
from unidecode import unidecode



def computeOracle(nameEmailData, aliasToIdName):
	oracle = MyDict()
	indices = range(len(nameEmailData.keys()))
	
#	print aliasToIdName.keys()
	
	# Initially all false
	for idx1, idx2 in itertools.combinations(indices, 2):
		oracle[(idx1, idx2)] = 0
	# Reflexive matches are always true: 
	# any (alias,email) pair should be matched to itself
	for idx in indices:
		oracle[(idx, idx)] = 1
	# Look up the identity name for an alias
	# Match (alias,email) pairs that share the same identity name
	for idx1, idx2 in itertools.combinations(indices, 2):
#		try:
		(name1, email1) = nameEmailData[idx1]
		(name2, email2) = nameEmailData[idx2]
#		print name1
#		if name1 not in aliasToIdName.keys():
#			print name1
#			exit()
			
		if aliasToIdName[name1] == aliasToIdName[name2]:
			oracle[(idx1, idx2)] = 1
#		except:
#			print idx1, nameEmailData[idx1]
#			print idx2, nameEmailData[idx2]
			
	return oracle	


def unspam(email):
	'''Normalize an email address'''
	if email.find('@') == -1:
		# gerard dot b at bbox dot fr
		email = email.replace(" dot ", ".")
		email = email.replace(" DOT ", ".")
		email = email.replace(" @ ", "@")
		email = email.replace(" at ", "@")
		email = email.replace(" AT ", "@")
		email = email.replace(" -at- ", "@")
		email = email.replace(" -dot- ", ".")
		email = email.replace(" gmail com", "@gmail.com")
		email = email.replace(".gmail.com", "@gmail.com")
		email = email.replace("%40", "@") # PHP
		email = email.replace(" cpan org", "@cpan.org")
		email = email.replace(" onirica com", "@onirica.com")
		email = email.replace(".wipro.com", "@wipro.com")
		email = email.replace(".163.com", "@163.com")
		email = email.replace(" thurman org uk", "@thurman.org.uk")
		email = email.replace(".gnome.org", "@gnome.org")
		email = email.replace(" zip com au", "@zip.com.au")
		email = email.replace(" gmail.com", "@gmail.com")
		email = email.replace("set EMAIL_ADDRESS environment variable", "")
		email = email.replace("tec.ifilp.org/info?gcompris", "")
		email = email.replace("Stefan Walter", "")
		email = email.replace("@redhat,^B", "@redhat.com")
		email = email.replace("@redhat.ocm", "@redhat.com")
	return email.strip().lower()


def strip_accents(s):
	return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
	
ignorechars = ".?,;:'\"!\/-_#~`&%$@*-+()_="
	
def normalize(s):
	s = unicode(s)
	return unidecode(''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')).lower()).translate(None, ignorechars).lower().strip().split()

def simpleNormalize(s):
	return unidecode(''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')).lower()).lower().strip()

	
def slugify(value):
	'''
	Normalizes string, converts to lowercase, removes non-alpha characters,
	and converts spaces to hyphens.
	'''
	try:
		value = unicodedata.normalize('NFKD', unicode(value, encoding='utf_8_sig'))#.encode('ascii', 'ignore')
		value = strip_accents(value)
		value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
		re.sub('[-\s]+', '-', value)
		value = value.replace('  ',' ')
		value = value.strip()
	except:
		print value
		print unicode(value, errors='ignore')
		v = unicode(value, encoding='utf_8_sig')
		# v = unicode(re.sub('[^\w\s-]', '', v).strip().lower())
		print strip_accents(v)
		
		exit()
	
	return value


def LevenshteinDistance(s, t):
	m = len(s)
	n = len(t)
	# for all i and j, d[i,j] will hold the Levenshtein distance between
	# the first i characters of s and the first j characters of t;
	# note that d has (m+1)x(n+1) values
	d = {}

	for i in range(m+1):
		d[i] = {}
		d[i][0] = i # the distance of any first string to an empty second string
	for j in range(n+1):
		d[0][j] = j # the distance of any second string to an empty first string

	for j in range(1,n+1):
		for i in range(1,m+1):
			if s[i-1] == t[j-1]:
				d[i][j] = d[i-1][j-1]       # no operation required
			else:
				#first: deletion
				#second: insertion
				#third: substitution
				d[i][j] = min((d[i-1][j] + 1), (d[i][j-1] + 1), (d[i-1][j-1] + 1))

	return d[m][n]



def similar(str1, str2, t):
	try:
		similarity = float(1) - (float(LevenshteinDistance(str1, str2)) / float(max(len(str1), len(str2))))
		if similarity >= t:
			return True
		else:
			return False
	except:
		return False


# Decode quoted-printable
def qpdecode(text):
	import email
	try:
		parts = email.Header.decode_header(text)
		new_header = email.Header.make_header(parts)
		human_readable = unicode(new_header)
		return human_readable
	except:
		return None
	
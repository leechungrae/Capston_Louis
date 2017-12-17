//  Backpanel P20 with 5 braille cells
//  -----------------------------------------------------------------------------------------------             7 +200V  <-------------------------- DC-DC +185V
// |  cell 0   |   cell 1  |  cell 2   |  cell 3   |  cell 4   |  cell 5   |  cell 6   |  cell 7   | <-- Cable  6 n.c.
//  ------------------------------------------------------------------------------------------------            5 GND    <----   Arduino GND    ---> DC-DC GND
// |  1 oo 8   |  1 oo 8   |  1 oo 8   |  1 oo 8   |  1 oo 8   |  values of the pins               |            4 CLK    <----   Arduino pin 5
// |  2 oo 16  |  2 oo 16  |  2 oo 16  |  2 oo 16  |  2 oo 16  |                                   |            3 STRB   <----   Arduino pin 4
// |  4 oo 32  |  4 oo 32  |  4 oo 32  |  4 oo 32  |  4 oo 32  |                                   |            2 Din    <----   Arduino pin 3
// | 64 oo 128 | 64 oo 128 | 64 oo 128 | 64 oo 128 | 64 oo 128 |                                   |            1 +5V    <----   Arduino +5V    ---> DC-DC +5V
//  -----------------------------------------------------------------------------------------------                              Arduino pin 2  ---> DC-DC /ON
//
// YouTube video: https://www.youtube.com/watch?v=3zqth08wQBs
// P20 cell:      http://web.metec-ag.de/P20.pdf
// DC-DC:         http://web.metec-ag.de/DCDC%20Converter%205to200V.pdf
// Contact:       ab@metec-ag.de

const int ON = 2;  
const int DATA = 3;
const int STROBE = 4;  
const int CLOCK = 5; 
const int BTN = 6;
const int POTI = 0;
  
const int cellCount = 8;
byte cells[cellCount];
String words[10];
String copyWords[10];

boolean first=true;

void setup()
{
  Serial.begin(9600); // 라즈베리파이와 시리얼 통신을 위해 setup()부분에 삽입
  Serial.setTimeout(3000);
  // 점자모듈로 출력하는 핀 세팅
  pinMode(ON, OUTPUT);
  pinMode(DATA, OUTPUT);
  pinMode(STROBE, OUTPUT);
  pinMode(CLOCK, OUTPUT);
  pinMode(BTN, INPUT_PULLUP);
  Reset();

  digitalWrite(ON, 0);  // 0=ON, 1=OFF
}

void loop() {

  if(first==true) {
    while(digitalRead(BTN)==1) {
          delay(1);
          if(digitalRead(BTN)==0) {
            break;
          }
        }

        while(digitalRead(BTN)==0) {
          delay(1);
          if(digitalRead(BTN)==1) {
            break;
          }
        }

        Serial.write("s");
        first=false;
  }
  
   char temp[100];
   char save[100];
   byte leng;
   char type[5];
char ch_len[2];
int int_len;

  if(Serial.available()){
    leng = Serial.readBytesUntil('!', temp, 50);
    temp[leng]='\0';
  }

  //Serial.print(temp);
  
  if (leng!=0) {  
    strncpy(type, temp, 4);
    type[4]='\0';
     

     if(strcmp(type,"0x07")==0) {
      
     strncpy(ch_len, temp+4,2);
     int_len=atoi(ch_len);
     strncpy(temp,temp+6,int_len);
     temp[int_len]='\0';
     strcpy(save,temp);
     
     char* result;
     int i=0, j=0, k=0, value=0;
     byte bValue=0;
     reset:
     result=strtok(temp, ";");
     
     while(result!=NULL)
     {
        words[i++]=result;
        result=strtok(NULL, ";"); // NULL이 들어오면 이전에 자기가 기억한 곳으로부터 분리를 시도
     }
     // 마지막에는 i가 저장한 배열의 수보다 1만큼 클
  
    for(j=0;j<i;j++) {
          
      if((j%2==0&&j!=0)) { // 두 음절씩 표현할수있으므로 대기, 2단어 정보를 입력했거나 마지막 단어를 기입하였을 시에
        //Serial.print("flush\n");
        Flush(); Wait();
        //delay(3000);
        k=0;
        // 버튼 대기
        //Serial.print("button:");
        //Serial.println(BTN);
        while(digitalRead(BTN)==1) {
   
          delay(1);
          if(digitalRead(BTN)==0) {
            break;
          }
        }

        while(digitalRead(BTN)==0) {
          //Serial.print("ok\n");
          delay(1);
          if(digitalRead(BTN)==1) {
            break;
          }
        }
        Reset();  
      }
     
      strcpy(temp, words[j].c_str());
    
      result=strtok(temp, ",");
      while(result!=NULL) 
      {
        value=atoi(result); // char 값을 int로
    
        bValue=(byte)(value&0xFF); // int를 byte로
  
        cells[k++]=bValue;
        //Serial.print(k);
        result=strtok(NULL,",");
      }

       // 이부분을 넣지않으면 2음절씩 이어서 나옴
        if(k%4!=0) { // 한 음절이 4자리를 차지하지않을때, 4자리를 차지할때까지 0값을 삽입
          //Serial.print("k\n");
          while(k%4!=0) {
            cells[k++]=0;
          }
        }
        
      if(j==i-1) { // 두 음절씩 표현할수있으므로 대기, 2단어 정보를 입력했거나 마지막 단어를 기입하였을 시에
        unsigned long start_time=0;
        //Serial.print("flush\n");
        Flush(); Wait();
        //delay(3000);
        //Reset();
        k=0;


        // 0으로 초기화
        while(digitalRead(BTN)==1) {
          delay(1);
          if(digitalRead(BTN)==0) {
            break;
          }
        }

        while(digitalRead(BTN)==0) {
          delay(1);
          if(digitalRead(BTN)==1) {
            break;
          }
        }

        Reset();
        


        // 다 내려간 상태에서 다시 띄울지, 종료신호를 보낼ㅈ
        // 버튼 대기
        while(digitalRead(BTN)==1) {
          delay(1);
          if(digitalRead(BTN)==0) {
            start_time=millis(); // 버튼이 눌렸을때부터 시간체크
            //Serial.print("time:");
            //Serial.println(start_time); //32756
            break;
          }
        }

        

        while(digitalRead(BTN)==0) {
          delay(1);
          if(digitalRead(BTN)==1) {
            break;
          }
        }
        Reset();
        if(start_time>0) {
        if(millis()-start_time>1200) {
          //Serial.println(millis()-start_time);
          Serial.write("f");
        } else {
          //Serial.println(millis()-start_time);
          i=0;
          j=0;
          strcpy(temp,save);
          goto reset;
        }
        }
        
        
      }
     }
     }

     else if(strcmp(type,"0x08")==0) {
       while(digitalRead(BTN)==1) {
          delay(1);
          if(digitalRead(BTN)==0) {
            break;
          }
        }

        while(digitalRead(BTN)==0) {
          delay(1);
          if(digitalRead(BTN)==1) {
            break;
          }
        }
        Serial.write("w");
     }

     else if(strcmp(type,"0x3e")==0) {
        digitalWrite(ON, 1); // DC cell power in a sleeping mode 
     }
     else if(strcmp(type, "0x3f")==0) {
       digitalWrite(ON, 0); // DC cell power on
     }
     
  
  }
}


void Wait()
{
  // optional. read the value from the poti on analog pin 0
  delay(analogRead(POTI) / 2);
}

void Reset()
{
  cells[0] = 0; cells[1] = 0; cells[2] = 0; cells[3] = 0;cells[4] = 0; cells[5]=0; cells[6]=0; cells[7]=0;
  Flush(); Wait();
}



// Send the data
void Flush()
{
  // This example is for one P20 backpanel. If you are using others you have to change the bit order!
  // P20: 6,7,2,1,0,5,4,3 P20은 이순서대로 비트사용(제작사인 METEC에서 정한 순서)
  // P16: 6,2,1,0,7,5,4,3
  // B11: 0,1,2,3,4,5,6,7  
  // Rotate the bytes from the right side. Cell 0 comes first and is on the left position.

  for (int i = 0; i < cellCount; i++) // 백패널에 부착된 모든 점자판에 값 세팅
  {
    if (bitRead(cells[i], 6)) { digitalWrite(DATA, 0); } // 6 자리의 비트를 읽어서 반환, 0을 보내서 점자가 올라오게함
    else { digitalWrite(DATA, 1); }     
    digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0); // DATA SEQUENCE(순서)를 다음으로 넘겨주기위해 CLOCK 사용
    if (bitRead(cells[i], 7)) { digitalWrite(DATA, 0); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
    if (bitRead(cells[i], 2)) { digitalWrite(DATA, 0
    ); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
    if (bitRead(cells[i], 1)) { digitalWrite(DATA, 0); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
    if (bitRead(cells[i], 0)) { digitalWrite(DATA, 0); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
    if (bitRead(cells[i], 5)) { digitalWrite(DATA, 0); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
    if (bitRead(cells[i], 4)) { digitalWrite(DATA, 0); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
    if (bitRead(cells[i], 3)) { digitalWrite(DATA, 0); }
    else { digitalWrite(DATA, 1); }     digitalWrite(CLOCK, 1); digitalWrite(CLOCK, 0);
  }

  digitalWrite(STROBE, 1);  // Strobe on 수신준비시킨후에 데이터 전송하는부분
  digitalWrite(STROBE, 0);  // Strobe off
}




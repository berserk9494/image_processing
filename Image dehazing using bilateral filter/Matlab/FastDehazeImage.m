classdef FastDehazeImage < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure              matlab.ui.Figure
        GridLayout            matlab.ui.container.GridLayout
        LeftPanel             matlab.ui.container.Panel
        Image                 matlab.ui.control.Image
        JSwitch               matlab.ui.control.Switch
        JSwitchLabel          matlab.ui.control.Label
        Adapt_EQSwitch        matlab.ui.control.Switch
        Adapt_EQSwitchLabel   matlab.ui.control.Label
        tSwitch               matlab.ui.control.Switch
        tSwitchLabel          matlab.ui.control.Label
        RSwitch               matlab.ui.control.Switch
        RSwitchLabel          matlab.ui.control.Label
        Sigma_r               matlab.ui.control.Slider
        Sigma_rSlider_2Label  matlab.ui.control.Label
        Sigma_t               matlab.ui.control.Slider
        Sigma_tSliderLabel    matlab.ui.control.Label
        RunButton             matlab.ui.control.Button
        V_RSwitch             matlab.ui.control.Switch
        V_RSwitchLabel        matlab.ui.control.Label
        VSwitch               matlab.ui.control.Switch
        VSwitchLabel          matlab.ui.control.Label
        WSwitch               matlab.ui.control.Switch
        WSwitchLabel          matlab.ui.control.Label
        pEditField            matlab.ui.control.NumericEditField
        pEditFieldLabel       matlab.ui.control.Label
        wEditField            matlab.ui.control.NumericEditField
        wEditFieldLabel       matlab.ui.control.Label
        OmegaSpinner          matlab.ui.control.Spinner
        OmegaSpinnerLabel     matlab.ui.control.Label
        KernelSpinner         matlab.ui.control.Spinner
        KernelSpinnerLabel    matlab.ui.control.Label
        SelectImageButton     matlab.ui.control.Button
        RightPanel            matlab.ui.container.Panel
        UIAxes_2              matlab.ui.control.UIAxes
        UIAxes                matlab.ui.control.UIAxes
    end

    % Properties that correspond to apps with auto-reflow
    properties (Access = private)
        onePanelWidth = 576;
    end

    
    properties (Access = private)
        imageData = []
        imageW = []
        imageV = []
        imageV_R = []
        imageR = []
        imaget = []
        imageJ = []
        imageJ_adapt = []
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: SelectImageButton
        function SelectImageButtonPushed(app, event)

            close all;

            [file ,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.gif;*.tif';'*.*'},'File Selector');
            selectedfile = fullfile(path,file);
            
            % Check if a file was selected
            if isequal(file, 0) % No file selected
                msgbox('There is NO image selected');
            else
                % Read the image file
                app.imageData=imread(selectedfile);
                imshow(app.imageData, 'Parent', app.UIAxes);
                [height, width, ~] = size(app.imageData);
                app.UIAxes.XLim = [0 width];
                app.UIAxes.YLim = [0 height];
                app.KernelSpinner.Enable='on';app.KernelSpinnerLabel.Enable='on';
                app.OmegaSpinner.Enable='on';app.OmegaSpinnerLabel.Enable='on';
                app.Sigma_r.Enable='on';app.Sigma_rSlider_2Label.Enable='on';
                app.Sigma_t.Enable='on';app.Sigma_tSliderLabel.Enable='on';
                app.pEditField.Enable='on';app.pEditFieldLabel.Enable='on';
                app.wEditField.Enable='on';app.wEditFieldLabel.Enable='on';
                app.tSwitch.Enable='on';app.tSwitchLabel.Enable='on';
                app.V_RSwitch.Enable='on';app.V_RSwitchLabel.Enable='on';
                app.RSwitch.Enable='on';app.RSwitchLabel.Enable='on';
                app.VSwitch.Enable='on';app.VSwitchLabel.Enable='on';
                app.WSwitch.Enable='on';app.WSwitchLabel.Enable='on';
                app.JSwitch.Enable='on';app.JSwitchLabel.Enable='on';
                app.Adapt_EQSwitch.Enable='on';app.Adapt_EQSwitchLabel.Enable='on';
                app.RunButton.Enable='on';
            end
         
        end

        % Button pushed function: RunButton
        function RunButtonPushed(app, event)
            close all;
            [height, width, ~] = size(app.imageData);
            %%
            sigma_s=0.03*min(height,width);
            sigma_r=app.Sigma_r.Value;
            sigma_t=app.Sigma_t.Value;
            p=app.pEditField.Value;
            w=app.wEditField.Value;
            t0=0.3;
            radius = app.KernelSpinner.Value;
            omega(1:2) = app.OmegaSpinner.Value;
            
            %%
            app.imageW = double(min(app.imageData,[],3));
            if strcmp(app.WSwitch.Value,'On')
               figure,imshow(uint8(app.imageW)),title('W'); 
            end
            
            
            %%
            B = medfilt2(app.imageW,omega,'symmetric');
            C=B-medfilt2(abs(app.imageW-B),omega,'symmetric');
            app.imageV=max(min(p.*C,app.imageW),0);
            if strcmp(app.VSwitch.Value,'On')
               figure,imshow(uint8(app.imageV)),title('V');
            end   
            %%
            app.imageR=bilat_filter(app.imageW,radius,sigma_s,sigma_r);
            if strcmp(app.RSwitch.Value,'On')
               figure,imshow(uint8(app.imageR)),title('R');
            end   
       
            %%
            app.imageV_R=bilat_filter_joint(app.imageV,app.imageR,radius,sigma_s,sigma_r,sigma_t);
            if strcmp(app.V_RSwitch.Value,'On')
               figure,imshow(uint8(app.imageV_R)),title('V_{R}');
            end   
            %%
            A  = min([estimateAtmosphericLight(app.imageW), max(max(255-app.imageW))]);
            app.imaget=ones(height,width)-w*app.imageV_R/A;
            if strcmp(app.tSwitch.Value,'On')
               figure,imshow(app.imaget),title('depth image t');
            end
            image_double=double(app.imageData);
            app.imageJ=zeros(size(app.imageData));
            app.imageJ(:,:,1)=(image_double(:,:,1)-A)./max(app.imaget,t0)+A;
            app.imageJ(:,:,2)=(image_double(:,:,2)-A)./max(app.imaget,t0)+A;
            app.imageJ(:,:,3)=(image_double(:,:,3)-A)./max(app.imaget,t0)+A;
    
            imshow(uint8(app.imageJ), 'Parent', app.UIAxes_2);
            [height, width, ~] = size(app.imageData);
            app.UIAxes_2.XLim = [0 width];
            app.UIAxes_2.YLim = [0 height];
            if strcmp(app.JSwitch.Value,'On')
               figure,imshow(uint8(app.imageJ)),title('J');
               figure,imshow(uint8(app.imageData)),title('Original');
            end
            
            %%
            if strcmp(app.Adapt_EQSwitch.Value,'On')
               LAB = rgb2lab(uint8(app.imageJ)); 
               L = LAB(:,:,1)/100;
               L = adapthisteq(L,'NumTiles',[8 8],'ClipLimit',0.005);
               LAB(:,:,1) = L*100;
               app.imageJ_adapt = lab2rgb(LAB);
               figure,imshow(app.imageJ_adapt),title('J_{adaptEQ}');
            end
            
        end

        % Changes arrangement of the app based on UIFigure width
        function updateAppLayout(app, event)
            currentFigureWidth = app.UIFigure.Position(3);
            if(currentFigureWidth <= app.onePanelWidth)
                % Change to a 2x1 grid
                app.GridLayout.RowHeight = {516, 516};
                app.GridLayout.ColumnWidth = {'1x'};
                app.RightPanel.Layout.Row = 2;
                app.RightPanel.Layout.Column = 1;
            else
                % Change to a 1x2 grid
                app.GridLayout.RowHeight = {'1x'};
                app.GridLayout.ColumnWidth = {245, '1x'};
                app.RightPanel.Layout.Row = 1;
                app.RightPanel.Layout.Column = 2;
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 633 516];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {245, '1x'};
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 0;
            app.GridLayout.Padding = [0 0 0 0];
            app.GridLayout.Scrollable = 'on';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create SelectImageButton
            app.SelectImageButton = uibutton(app.LeftPanel, 'push');
            app.SelectImageButton.ButtonPushedFcn = createCallbackFcn(app, @SelectImageButtonPushed, true);
            app.SelectImageButton.Position = [67 488 100 22];
            app.SelectImageButton.Text = 'Select Image';

            % Create KernelSpinnerLabel
            app.KernelSpinnerLabel = uilabel(app.LeftPanel);
            app.KernelSpinnerLabel.HorizontalAlignment = 'right';
            app.KernelSpinnerLabel.Enable = 'off';
            app.KernelSpinnerLabel.Position = [12 452 40 22];
            app.KernelSpinnerLabel.Text = 'Kernel';

            % Create KernelSpinner
            app.KernelSpinner = uispinner(app.LeftPanel);
            app.KernelSpinner.Step = 2;
            app.KernelSpinner.Limits = [1 50];
            app.KernelSpinner.RoundFractionalValues = 'on';
            app.KernelSpinner.Enable = 'off';
            app.KernelSpinner.Position = [67 452 100 22];
            app.KernelSpinner.Value = 15;

            % Create OmegaSpinnerLabel
            app.OmegaSpinnerLabel = uilabel(app.LeftPanel);
            app.OmegaSpinnerLabel.HorizontalAlignment = 'right';
            app.OmegaSpinnerLabel.Enable = 'off';
            app.OmegaSpinnerLabel.Position = [7 423 45 22];
            app.OmegaSpinnerLabel.Text = 'Omega';

            % Create OmegaSpinner
            app.OmegaSpinner = uispinner(app.LeftPanel);
            app.OmegaSpinner.Step = 2;
            app.OmegaSpinner.Limits = [1 50];
            app.OmegaSpinner.RoundFractionalValues = 'on';
            app.OmegaSpinner.Enable = 'off';
            app.OmegaSpinner.Position = [67 423 100 22];
            app.OmegaSpinner.Value = 15;

            % Create wEditFieldLabel
            app.wEditFieldLabel = uilabel(app.LeftPanel);
            app.wEditFieldLabel.HorizontalAlignment = 'right';
            app.wEditFieldLabel.Enable = 'off';
            app.wEditFieldLabel.Position = [14 282 25 22];
            app.wEditFieldLabel.Text = 'w';

            % Create wEditField
            app.wEditField = uieditfield(app.LeftPanel, 'numeric');
            app.wEditField.Limits = [0.5 1];
            app.wEditField.Enable = 'off';
            app.wEditField.Position = [54 282 39 22];
            app.wEditField.Value = 0.95;

            % Create pEditFieldLabel
            app.pEditFieldLabel = uilabel(app.LeftPanel);
            app.pEditFieldLabel.HorizontalAlignment = 'right';
            app.pEditFieldLabel.Enable = 'off';
            app.pEditFieldLabel.Position = [140 282 25 22];
            app.pEditFieldLabel.Text = 'p';

            % Create pEditField
            app.pEditField = uieditfield(app.LeftPanel, 'numeric');
            app.pEditField.Limits = [0.5 1];
            app.pEditField.Enable = 'off';
            app.pEditField.Position = [180 282 39 22];
            app.pEditField.Value = 0.95;

            % Create WSwitchLabel
            app.WSwitchLabel = uilabel(app.LeftPanel);
            app.WSwitchLabel.HorizontalAlignment = 'center';
            app.WSwitchLabel.Enable = 'off';
            app.WSwitchLabel.Position = [9 250 25 22];
            app.WSwitchLabel.Text = 'W';

            % Create WSwitch
            app.WSwitch = uiswitch(app.LeftPanel, 'slider');
            app.WSwitch.Orientation = 'vertical';
            app.WSwitch.Enable = 'off';
            app.WSwitch.Position = [12 179 20 45];

            % Create VSwitchLabel
            app.VSwitchLabel = uilabel(app.LeftPanel);
            app.VSwitchLabel.HorizontalAlignment = 'center';
            app.VSwitchLabel.Enable = 'off';
            app.VSwitchLabel.Position = [46 250 25 22];
            app.VSwitchLabel.Text = 'V';

            % Create VSwitch
            app.VSwitch = uiswitch(app.LeftPanel, 'slider');
            app.VSwitch.Orientation = 'vertical';
            app.VSwitch.Enable = 'off';
            app.VSwitch.Position = [50 179 20 45];

            % Create V_RSwitchLabel
            app.V_RSwitchLabel = uilabel(app.LeftPanel);
            app.V_RSwitchLabel.HorizontalAlignment = 'center';
            app.V_RSwitchLabel.Enable = 'off';
            app.V_RSwitchLabel.Position = [118 251 29 22];
            app.V_RSwitchLabel.Text = 'V_R';

            % Create V_RSwitch
            app.V_RSwitch = uiswitch(app.LeftPanel, 'slider');
            app.V_RSwitch.Orientation = 'vertical';
            app.V_RSwitch.Enable = 'off';
            app.V_RSwitch.Position = [123 179 20 45];

            % Create RunButton
            app.RunButton = uibutton(app.LeftPanel, 'push');
            app.RunButton.ButtonPushedFcn = createCallbackFcn(app, @RunButtonPushed, true);
            app.RunButton.Enable = 'off';
            app.RunButton.Position = [103 46 127 85];
            app.RunButton.Text = 'Run';

            % Create Sigma_tSliderLabel
            app.Sigma_tSliderLabel = uilabel(app.LeftPanel);
            app.Sigma_tSliderLabel.HorizontalAlignment = 'right';
            app.Sigma_tSliderLabel.Enable = 'off';
            app.Sigma_tSliderLabel.Position = [8 342 50 22];
            app.Sigma_tSliderLabel.Text = 'Sigma_t';

            % Create Sigma_t
            app.Sigma_t = uislider(app.LeftPanel);
            app.Sigma_t.Limits = [0.01 100];
            app.Sigma_t.MajorTicks = [0.01 20 40 60 80 100];
            app.Sigma_t.Enable = 'off';
            app.Sigma_t.Position = [79 351 150 3];
            app.Sigma_t.Value = 20;

            % Create Sigma_rSlider_2Label
            app.Sigma_rSlider_2Label = uilabel(app.LeftPanel);
            app.Sigma_rSlider_2Label.HorizontalAlignment = 'right';
            app.Sigma_rSlider_2Label.Enable = 'off';
            app.Sigma_rSlider_2Label.Position = [8 396 50 22];
            app.Sigma_rSlider_2Label.Text = 'Sigma_r';

            % Create Sigma_r
            app.Sigma_r = uislider(app.LeftPanel);
            app.Sigma_r.Limits = [0.01 100];
            app.Sigma_r.MajorTicks = [0.01 20 40 60 80 100];
            app.Sigma_r.Enable = 'off';
            app.Sigma_r.Position = [79 405 150 3];
            app.Sigma_r.Value = 20;

            % Create RSwitchLabel
            app.RSwitchLabel = uilabel(app.LeftPanel);
            app.RSwitchLabel.HorizontalAlignment = 'center';
            app.RSwitchLabel.Enable = 'off';
            app.RSwitchLabel.Position = [84 251 25 22];
            app.RSwitchLabel.Text = 'R';

            % Create RSwitch
            app.RSwitch = uiswitch(app.LeftPanel, 'slider');
            app.RSwitch.Orientation = 'vertical';
            app.RSwitch.Enable = 'off';
            app.RSwitch.Position = [85 179 23 45];

            % Create tSwitchLabel
            app.tSwitchLabel = uilabel(app.LeftPanel);
            app.tSwitchLabel.HorizontalAlignment = 'center';
            app.tSwitchLabel.Enable = 'off';
            app.tSwitchLabel.Position = [157 251 25 22];
            app.tSwitchLabel.Text = 't';

            % Create tSwitch
            app.tSwitch = uiswitch(app.LeftPanel, 'slider');
            app.tSwitch.Orientation = 'vertical';
            app.tSwitch.Enable = 'off';
            app.tSwitch.Position = [160 179 20 45];

            % Create Adapt_EQSwitchLabel
            app.Adapt_EQSwitchLabel = uilabel(app.LeftPanel);
            app.Adapt_EQSwitchLabel.HorizontalAlignment = 'center';
            app.Adapt_EQSwitchLabel.Enable = 'off';
            app.Adapt_EQSwitchLabel.Position = [38 77 61 22];
            app.Adapt_EQSwitchLabel.Text = 'Adapt_EQ';

            % Create Adapt_EQSwitch
            app.Adapt_EQSwitch = uiswitch(app.LeftPanel, 'slider');
            app.Adapt_EQSwitch.Orientation = 'vertical';
            app.Adapt_EQSwitch.Enable = 'off';
            app.Adapt_EQSwitch.Position = [12 66 20 45];

            % Create JSwitchLabel
            app.JSwitchLabel = uilabel(app.LeftPanel);
            app.JSwitchLabel.HorizontalAlignment = 'center';
            app.JSwitchLabel.Enable = 'off';
            app.JSwitchLabel.Position = [194 251 25 22];
            app.JSwitchLabel.Text = 'J';

            % Create JSwitch
            app.JSwitch = uiswitch(app.LeftPanel, 'slider');
            app.JSwitch.Orientation = 'vertical';
            app.JSwitch.Enable = 'off';
            app.JSwitch.Position = [197 179 20 45];

            % Create Image
            app.Image = uiimage(app.LeftPanel);
            app.Image.Position = [8 1 232 41];
            app.Image.ImageSource = fullfile(pathToMLAPP, '2023-05-24_22h40_17.png');

            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create UIAxes
            app.UIAxes = uiaxes(app.RightPanel);
            title(app.UIAxes, 'Original')
            app.UIAxes.XColor = 'none';
            app.UIAxes.XTick = [];
            app.UIAxes.YColor = 'none';
            app.UIAxes.YTick = [];
            app.UIAxes.ZColor = 'none';
            app.UIAxes.GridColor = 'none';
            app.UIAxes.MinorGridColor = 'none';
            app.UIAxes.Position = [24 260 338 250];

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.RightPanel);
            title(app.UIAxes_2, 'Dehazed (J)')
            app.UIAxes_2.XColor = 'none';
            app.UIAxes_2.XTick = [];
            app.UIAxes_2.YColor = 'none';
            app.UIAxes_2.YTick = [];
            app.UIAxes_2.ZColor = 'none';
            app.UIAxes_2.GridColor = [0.15 0.15 0.15];
            app.UIAxes_2.MinorGridColor = 'none';
            app.UIAxes_2.Position = [24 7 338 250];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = FastDehazeImage

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end